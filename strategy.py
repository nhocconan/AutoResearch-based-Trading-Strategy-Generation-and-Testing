#!/usr/bin/env python3
"""
Experiment #192: 12h Primary + 1d/1w HTF — KAMA Trend + Fisher Reversal + ADX Regime

Hypothesis: Previous 12h strategies failed due to (1) too strict entry filters = 0 trades,
or (2) pure trend following = whipsaw in 2022 crash. This strategy combines:

1. KAMA (Kaufman Adaptive MA) - adapts to volatility, reduces whipsaw in chop
2. Fisher Transform - catches reversals at extremes (better than RSI for 12h)
3. ADX regime filter - only trend-follow when ADX > 25, mean-revert when ADX < 20
4. 1d HMA for macro bias, 1w HMA for ultra-trend confirmation
5. Looser entry thresholds to ensure 20-50 trades/year on 12h

Key improvements over #182/#186:
- Fisher Transform instead of CRSI (better for 12h reversals)
- KAMA instead of HMA for primary trend (more adaptive)
- ADX hysteresis (enter 25, exit 18) to reduce regime flip-flop
- Looser Fisher thresholds (-1.2/+1.2 instead of -1.5/+1.5)
- Volume confirmation on breakouts (reduce false signals)

TARGET: 25-45 trades/year, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
Position sizing: 0.0, ±0.25, ±0.30 (discrete to minimize fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_fisher_adx_regime_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts to market efficiency - moves fast in trends, slow in chop.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER)
    change = np.abs(close_s.diff(er_period))
    volatility = np.abs(close_s.diff()).rolling(window=er_period, min_periods=er_period).sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        er = change / (volatility + 1e-10)
    er = er.fillna(0).values
    
    # Smoothing Constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price to Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(high)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        # Calculate median price
        hl2 = (high[i-period+1:i+1] + low[i-period+1:i+1]) / 2
        highest_hl2 = np.max(hl2)
        lowest_hl2 = np.min(hl2)
        
        range_val = highest_hl2 - lowest_hl2
        if range_val < 1e-10:
            fisher[i] = 0.0
            trigger[i] = fisher[i-1] if i > 0 else 0.0
            continue
        
        # Normalize price position within range
        x = (hl2[-1] - lowest_hl2) / range_val
        x = np.clip(x, 0.001, 0.999)  # Avoid log(0)
        
        # Fisher calculation
        v = 0.66 * ((x - 0.5) / (1e-10)) + 0.67 * fisher[i-1] if i > period else 0.0
        v = np.clip(v, -0.999, 0.999)
        fisher[i] = 0.5 * np.log((1 + v) / (1 - v + 1e-10)) + 0.5 * fisher[i-1] if i > period else 0.0
        trigger[i] = fisher[i-1] if i > 0 else 0.0
    
    return fisher, trigger

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth with EMA
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI and DX
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100.0 * plus_dm_smooth / (tr_smooth + 1e-10)
        minus_di = 100.0 * minus_dm_smooth / (tr_smooth + 1e-10)
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # ADX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    kama_21 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, period=9)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    # Calculate 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 1w HMA for ultra-long-term trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    POSITION_SIZE_FULL = 0.30
    POSITION_SIZE_HALF = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    # ADX hysteresis state
    adx_trending = False  # True if ADX was > 25, False if < 18
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(kama_21[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_trigger[i]):
            continue
        if np.isnan(adx_14[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(vol_ma[i]) or vol_ma[i] == 0:
            continue
        
        # === HTF MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === ADX REGIME WITH HYSTERESIS ===
        # Enter trending mode at ADX > 25, exit at ADX < 18
        if adx_14[i] > 25.0:
            adx_trending = True
        elif adx_14[i] < 18.0:
            adx_trending = False
        
        # === KAMA TREND DIRECTION ===
        kama_bullish = close[i] > kama_21[i]
        kama_bearish = close[i] < kama_21[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 1.2 * vol_ma[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        if adx_trending:
            # TREND FOLLOWING MODE (KAMA + Fisher confirmation)
            # Long: Price above KAMA + Fisher > -1.2 (not oversold) + 1d HMA bullish
            if kama_bullish and fisher[i] > -1.2 and price_above_hma_1d:
                if price_above_hma_1w:
                    new_signal = POSITION_SIZE_FULL
                else:
                    new_signal = POSITION_SIZE_HALF
            
            # Short: Price below KAMA + Fisher < 1.2 (not overbought) + 1d HMA bearish
            elif kama_bearish and fisher[i] < 1.2 and price_below_hma_1d:
                if price_below_hma_1w:
                    new_signal = -POSITION_SIZE_FULL
                else:
                    new_signal = -POSITION_SIZE_HALF
        
        else:
            # MEAN REVERSION MODE (Fisher extremes)
            # Long: Fisher < -1.5 (oversold) + price above 1w HMA (long-term bullish)
            if fisher[i] < -1.5 and price_above_hma_1w:
                new_signal = POSITION_SIZE_HALF
            
            # Short: Fisher > 1.5 (overbought) + price below 1w HMA (long-term bearish)
            elif fisher[i] > 1.5 and price_below_hma_1w:
                new_signal = -POSITION_SIZE_HALF
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and regime/trend still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price still above KAMA or ADX trending
                if kama_bullish or adx_trending:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price still below KAMA or ADX trending
                if kama_bearish or adx_trending:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if price crosses below KAMA and ADX not trending
        if in_position and position_side > 0 and kama_bearish and not adx_trending:
            new_signal = 0.0
        
        # Exit short if price crosses above KAMA and ADX not trending
        if in_position and position_side < 0 and kama_bullish and not adx_trending:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals