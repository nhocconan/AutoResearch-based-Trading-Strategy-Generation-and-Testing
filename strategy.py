#!/usr/bin/env python3
"""
Experiment #274: 4h Primary + 12h/1d HTF — Regime-Adaptive Dual Strategy

Hypothesis: Previous 4h strategies failed because they used ONE logic for all market conditions.
This version ADAPTS to regime:
- TREND REGIME (ADX>25, CHOP<38): Donchian breakout + HMA trend confirmation
- RANGE REGIME (ADX<20, CHOP>61): RSI mean reversion at Bollinger extremes
- TRANSITION (between): Stay flat, avoid whipsaws

KEY INSIGHTS from failed experiments:
- #269 (BB squeeze): 0 trades - too strict
- #271 (Chop+CRSI): Sharpe=-1.881 - over-filtered
- Current best uses CRSI+Chop but we need DIFFERENT combination

NEW APPROACH:
- 12h HMA(21) for MACRO bias (long only above, short only below)
- 4h ADX(14) + Choppiness(14) for REGIME detection
- 4h Donchian(20) for trend breakouts
- 4h BB(20,2) + RSI(14) for range mean-reversion
- ATR(14) 2.5x trailing stoploss
- Position size: 0.28 (balanced for 4h volatility)

TARGET: 25-45 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_adaptive_donchian_bb_12h1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.clip(lower=0)
    minus_dm = minus_dm.clip(lower=0)
    
    # True DM logic
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    highest_high = high_s.rolling(window=period, min_periods=period).max()
    lowest_low = low_s.rolling(window=period, min_periods=period).min()
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    sum_tr = pd.Series(tr).rolling(window=period, min_periods=period).sum()
    range_hl = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_tr / (range_hl + 1e-10)) / np.log10(period)
    
    return chop.fillna(50.0).values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper.values, lower.values, sma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align 12h HMA for macro bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA for broader trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.28  # Conservative for 4h volatility
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(donch_upper[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        # Trend regime: ADX > 25 AND Choppiness < 38
        is_trend_regime = (adx_14[i] > 25.0) and (chop_14[i] < 38.0)
        
        # Range regime: ADX < 20 AND Choppiness > 61
        is_range_regime = (adx_14[i] < 20.0) and (chop_14[i] > 61.0)
        
        # Transition: neither trend nor range (stay flat)
        is_transition = not is_trend_regime and not is_range_regime
        
        # === MACRO BIAS (12h/1d HMA) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # === TREND REGIME: Donchian Breakout + HMA confirmation ===
        if is_trend_regime:
            # Long: Price breaks Donchian high + above 12h HMA + HMA21 > HMA50
            donchian_breakout_long = close[i] > donch_upper[i-1]  # Break above previous high
            hma_bullish = hma_21[i] > hma_50[i]
            
            if donchian_breakout_long and price_above_hma_12h and hma_bullish:
                desired_signal = POSITION_SIZE
            
            # Short: Price breaks Donchian low + below 12h HMA + HMA21 < HMA50
            donchian_breakout_short = close[i] < donch_lower[i-1]  # Break below previous low
            hma_bearish = hma_21[i] < hma_50[i]
            
            if donchian_breakout_short and price_below_hma_12h and hma_bearish:
                desired_signal = -POSITION_SIZE
        
        # === RANGE REGIME: Mean Reversion at BB extremes ===
        elif is_range_regime:
            # Long: Price at BB lower + RSI oversold + above 12h HMA (bias long in uptrend)
            at_bb_lower = close[i] <= bb_lower[i] * 1.005  # Within 0.5% of lower band
            rsi_oversold = rsi_14[i] < 35.0
            
            if at_bb_lower and rsi_oversold and price_above_hma_12h:
                desired_signal = POSITION_SIZE
            
            # Short: Price at BB upper + RSI overbought + below 12h HMA (bias short in downtrend)
            at_bb_upper = close[i] >= bb_upper[i] * 0.995  # Within 0.5% of upper band
            rsi_overbought = rsi_14[i] > 65.0
            
            if at_bb_upper and rsi_overbought and price_below_hma_12h:
                desired_signal = -POSITION_SIZE
        
        # === TRANSITION REGIME: Stay flat ===
        elif is_transition:
            desired_signal = 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === REGIME CHANGE EXIT ===
        # Exit long if regime changes from trend to range (or vice versa inappropriately)
        if in_position and position_side > 0:
            # Exit if trend regime ends and we're in transition
            if is_trend_regime == False and is_range_regime == False:
                desired_signal = 0.0
            # Exit if macro bias flips
            if price_below_hma_12h and hma_12h_aligned[i] < hma_12h_aligned[i-1] * 0.995:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if is_trend_regime == False and is_range_regime == False:
                desired_signal = 0.0
            if price_above_hma_12h and hma_12h_aligned[i] > hma_12h_aligned[i-1] * 1.005:
                desired_signal = 0.0
        
        # === RSI EXTREME TAKE PROFIT ===
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        # If already in position and no exit signal, maintain position
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if still in valid regime
                if is_trend_regime or (is_range_regime and price_above_hma_12h):
                    desired_signal = POSITION_SIZE
            elif position_side < 0:
                # Hold short if still in valid regime
                if is_trend_regime or (is_range_regime and price_below_hma_12h):
                    desired_signal = -POSITION_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals