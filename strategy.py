#!/usr/bin/env python3
"""
Experiment #800: 1h Primary + 4h/12h HTF — Fisher Transform + HMA Trend + ADX Regime

Hypothesis: After 500+ failed strategies and analyzing recent failures:
1. 1h timeframe needs VERY strict filters to avoid fee drag (target 30-60 trades/year)
2. Fisher Transform catches reversals better than RSI/CRSI in bear/range markets (2022, 2025)
3. 4h HMA(21) provides proven trend bias (from current best Sharpe=0.612 strategy)
4. 12h ADX(14) for regime: ADX>25=trend, ADX<20=range (hysteresis prevents churn)
5. Session filter REMOVED — caused 0 trades in #790, #795
6. Volume filter relaxed to 0.8x (not 1.3x) — ensures trade generation
7. Asymmetric sizing: 0.30 in trends, 0.20 in ranges (controls fee drag)
8. ATR(14) trailing stop at 2.0x for protection

Key differences from failed 1h strategies (#790, #795):
- Fisher Transform instead of CRSI (better reversal detection)
- NO session filter (was killing trade generation)
- Volume threshold 0.8x (not 1.5x)
- ADX hysteresis: enter at 25, exit at 18 (prevents regime churn)
- Relaxed Fisher thresholds: -1.5/+1.5 (not -2.0/+2.0)
- Hold logic: maintain position until opposite signal or stoploss

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 30-60 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_hma_adx_4h12h_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_ema(series, period):
    """Exponential Moving Average."""
    return pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform — normalizes price to Gaussian distribution.
    Catches reversals at extremes better than RSI.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, trigger
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = 0.0
            trigger[i] = 0.0
            continue
        
        # Normalize price to 0-1 range
        price = (high[i] + low[i]) / 2.0
        value = (price - lowest) / (highest - lowest)
        
        # Constrain to 0.001-0.999 to avoid log(0)
        value = np.clip(value, 0.001, 0.999)
        
        # Fisher calculation
        fisher[i] = 0.5 * np.log((1 + value) / (1 - value))
        
        # Trigger line (1-period lag of fisher)
        if i > period:
            trigger[i] = fisher[i - 1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index — measures trend strength.
    ADX > 25 = trending, ADX < 20 = ranging.
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth with EMA
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate +DI and -DI
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
        minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
    
    # Calculate DX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # ADX is EMA of DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands."""
    sma = calculate_sma(close, period)
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, lower, sma

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate primary (1h) indicators
    fisher_1h, fisher_trigger_1h = calculate_fisher_transform(high, low, period=9)
    atr_1h = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger(close, period=20, std_mult=2.0)
    vol_sma_1h = calculate_volume_sma(volume, period=20)
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h ADX for regime detection
    adx_12h_raw = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, period=14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h_raw)
    
    signals = np.zeros(n)
    TREND_SIZE = 0.30
    RANGE_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # ADX regime hysteresis tracking
    prev_adx_regime = None  # None, 'trend', or 'range'
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(fisher_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(bb_sma[i]):
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        if np.isnan(adx_12h_aligned[i]):
            continue
        if np.isnan(vol_sma_1h[i]) or vol_sma_1h[i] <= 1e-10:
            continue
        
        # === TREND BIAS (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (12h ADX with hysteresis) ===
        adx_value = adx_12h_aligned[i]
        
        if prev_adx_regime is None:
            # Initialize regime
            if adx_value > 25:
                prev_adx_regime = 'trend'
            elif adx_value < 20:
                prev_adx_regime = 'range'
            else:
                prev_adx_regime = 'neutral'
        else:
            # Hysteresis: only switch at thresholds
            if prev_adx_regime == 'trend' and adx_value < 18:
                prev_adx_regime = 'range'
            elif prev_adx_regime == 'range' and adx_value > 25:
                prev_adx_regime = 'trend'
        
        trending_regime = prev_adx_regime == 'trend'
        ranging_regime = prev_adx_regime == 'range'
        
        # === VOLUME CONFIRMATION (relaxed) ===
        volume_confirmed = volume[i] > 0.8 * vol_sma_1h[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_value = fisher_1h[i]
        fisher_prev = fisher_1h[i-1] if i > 0 else fisher_value
        fisher_trigger = fisher_trigger_1h[i]
        
        # Fisher crossover signals
        fisher_bullish_cross = fisher_value > -1.5 and fisher_prev <= -1.5
        fisher_bearish_cross = fisher_value < 1.5 and fisher_prev >= 1.5
        
        # Fisher extreme levels
        fisher_extreme_low = fisher_value < -2.0
        fisher_extreme_high = fisher_value > 2.0
        
        # === BOLLINGER POSITION ===
        below_bb_lower = close[i] < bb_lower[i]
        above_bb_upper = close[i] > bb_upper[i]
        
        desired_signal = 0.0
        
        # === TRENDING REGIME LOGIC (ADX > 25) ===
        if trending_regime:
            # Trend following long: 4h bullish + Fisher bullish cross + volume
            if trend_4h_bullish and fisher_bullish_cross:
                desired_signal = TREND_SIZE if volume_confirmed else RANGE_SIZE
            
            # Trend following short: 4h bearish + Fisher bearish cross + volume
            if trend_4h_bearish and fisher_bearish_cross:
                desired_signal = -TREND_SIZE if volume_confirmed else -RANGE_SIZE
            
            # Pullback entry in trend: Fisher extreme + trend alignment
            if trend_4h_bullish and fisher_extreme_low and volume_confirmed:
                desired_signal = RANGE_SIZE
            
            if trend_4h_bearish and fisher_extreme_high and volume_confirmed:
                desired_signal = -RANGE_SIZE
        
        # === RANGING REGIME LOGIC (ADX < 20) ===
        elif ranging_regime:
            # Mean reversion long: Fisher extreme low + below BB lower
            if fisher_extreme_low and below_bb_lower:
                desired_signal = RANGE_SIZE if volume_confirmed else 0.0
            
            # Mean reversion short: Fisher extreme high + above BB upper
            if fisher_extreme_high and above_bb_upper:
                desired_signal = -RANGE_SIZE if volume_confirmed else 0.0
            
            # Conservative: Fisher cross at BB bounds
            if fisher_bullish_cross and below_bb_lower:
                desired_signal = RANGE_SIZE
            
            if fisher_bearish_cross and above_bb_upper:
                desired_signal = -RANGE_SIZE
        
        # === NEUTRAL REGIME (20 <= ADX <= 25) ===
        else:
            # Conservative: only Fisher extremes with trend alignment
            if fisher_extreme_low and trend_4h_bullish:
                desired_signal = RANGE_SIZE
            
            if fisher_extreme_high and trend_4h_bearish:
                desired_signal = -RANGE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h trend bullish and Fisher not extreme high
                if trend_4h_bullish and fisher_value < 2.0:
                    desired_signal = TREND_SIZE if trending_regime else RANGE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend bearish and Fisher not extreme low
                if trend_4h_bearish and fisher_value > -2.0:
                    desired_signal = -TREND_SIZE if trending_regime else -RANGE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses bearish
            if trend_4h_bearish and fisher_value > 1.5:
                desired_signal = 0.0
            # Exit if price hits BB upper in ranging regime
            if ranging_regime and above_bb_upper and fisher_value > 0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses bullish
            if trend_4h_bullish and fisher_value < -1.5:
                desired_signal = 0.0
            # Exit if price hits BB lower in ranging regime
            if ranging_regime and below_bb_lower and fisher_value < 0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= TREND_SIZE:
                desired_signal = TREND_SIZE
            else:
                desired_signal = RANGE_SIZE
        elif desired_signal < 0:
            if desired_signal <= -TREND_SIZE:
                desired_signal = -TREND_SIZE
            else:
                desired_signal = -RANGE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals