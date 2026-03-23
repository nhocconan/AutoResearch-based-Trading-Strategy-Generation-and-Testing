#!/usr/bin/env python3
"""
Experiment #275: 1h Primary + 4h/1d HTF — Fisher Transform + ADX Regime

Hypothesis: Previous 1h strategies failed from weak entry signals (RSI pullback too common)
and poor regime detection (Choppiness didn't work). This version uses:
- 1d HMA(21) for MACRO trend direction (very slow, reduces whipsaws in bear markets)
- 4h ADX(14) for REGIME detection (ADX>25 = trend follow, ADX<20 = mean revert)
- 1h Fisher Transform(9) for ENTRY timing (catches reversals better than RSI in bear markets)
- ATR(14) 2.5x trailing stoploss
- Position size: 0.25 (conservative for 1h volatility)

KEY INSIGHTS from failures:
- #270: RSI 40-60 triggered too often → too many trades → fee drag
- #265/#266: Choppiness Index regime filter didn't work (negative Sharpe)
- Fisher Transform excels in bear/range markets (2022 crash, 2025 bear)
- 1d HMA provides strong macro bias to avoid counter-trend trades

Fisher Transform Logic:
- Long when Fisher crosses above -1.5 (oversold reversal)
- Short when Fisher crosses below +1.5 (overbought reversal)
- Only enter when aligned with 1d HMA trend and 4h ADX regime

TARGET: 40-70 trades/year on 1h, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_adx_regime_1d4h_atr_v1"
timeframe = "1h"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Converts price to Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Calculate typical price
    typical = (high_s + low_s) / 2.0
    
    # Normalize price to 0-1 range using Donchian channel
    highest = typical.rolling(window=period, min_periods=period).max()
    lowest = typical.rolling(window=period, min_periods=period).min()
    
    # Avoid division by zero
    range_val = highest - lowest
    range_val = range_val.replace(0, 1e-10)
    
    # Normalize to -1 to +1
    normalized = 2.0 * (typical - lowest) / range_val - 1.0
    
    # Apply exponential smoothing
    smoothed = normalized.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Fisher transform
    with np.errstate(divide='ignore', invalid='ignore'):
        fisher_raw = 0.5 * np.log((1.0 + smoothed) / (1.0 - smoothed + 1e-10))
    
    # Signal line (1-period lag of Fisher)
    fisher = pd.Series(fisher_raw).fillna(0.0)
    
    return fisher.values

def calculate_adx(high, low, close, period=14):
    """
    Calculate ADX (Average Directional Index).
    ADX > 25 = trending market, ADX < 20 = ranging market
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high_s - high_s.shift(1)
    down_move = low_s.shift(1) - low_s
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values using Wilder's smoothing (EMA with span=period)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100.0 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10)
    
    # DX and ADX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher_1h = calculate_fisher_transform(high, low, close, period=9)
    
    # Calculate and align 4h ADX for regime detection
    adx_4h_raw, plus_di_4h_raw, minus_di_4h_raw = calculate_adx(
        df_4h['high'].values, 
        df_4h['low'].values, 
        df_4h['close'].values, 
        period=14
    )
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h_raw)
    plus_di_4h_aligned = align_htf_to_ltf(prices, df_4h, plus_di_4h_raw)
    minus_di_4h_aligned = align_htf_to_ltf(prices, df_4h, minus_di_4h_raw)
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.25  # Conservative for 1h volatility
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track Fisher crosses for entry timing
    prev_fisher = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            prev_fisher = fisher_1h[i] if not np.isnan(fisher_1h[i]) else prev_fisher
            continue
        if np.isnan(fisher_1h[i]):
            signals[i] = 0.0
            prev_fisher = fisher_1h[i] if not np.isnan(fisher_1h[i]) else prev_fisher
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(adx_4h_aligned[i]):
            signals[i] = 0.0
            prev_fisher = fisher_1h[i] if not np.isnan(fisher_1h[i]) else prev_fisher
            continue
        
        current_fisher = fisher_1h[i]
        
        # === MACRO BIAS (1d HMA) - PRIMARY DIRECTION FILTER ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (4h ADX) ===
        adx_value = adx_4h_aligned[i]
        trending_regime = adx_value > 25.0  # ADX > 25 = trending
        ranging_regime = adx_value < 20.0   # ADX < 20 = ranging
        
        # === DIRECTIONAL BIAS (4h DI) ===
        di_bullish = plus_di_4h_aligned[i] > minus_di_4h_aligned[i]
        di_bearish = plus_di_4h_aligned[i] < minus_di_4h_aligned[i]
        
        # === FISHER TRANSFORM ENTRY SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_cross_long = (prev_fisher <= -1.5) and (current_fisher > -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_cross_short = (prev_fisher >= 1.5) and (current_fisher < 1.5)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY: 1d bullish + (4h trending + DI bullish OR 4h ranging) + Fisher cross
        if price_above_hma_1d:
            if trending_regime and di_bullish and fisher_cross_long:
                desired_signal = POSITION_SIZE
            elif ranging_regime and fisher_cross_long:
                desired_signal = POSITION_SIZE
        
        # SHORT ENTRY: 1d bearish + (4h trending + DI bearish OR 4h ranging) + Fisher cross
        elif price_below_hma_1d:
            if trending_regime and di_bearish and fisher_cross_short:
                desired_signal = -POSITION_SIZE
            elif ranging_regime and fisher_cross_short:
                desired_signal = -POSITION_SIZE
        
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
        
        # === TREND REVERSAL EXIT (1d HMA flip) ===
        if in_position and position_side > 0 and price_below_hma_1d:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_hma_1d:
            desired_signal = 0.0
        
        # === FISHER EXTREME EXIT (take profit) ===
        if in_position and position_side > 0 and current_fisher > 1.5:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and current_fisher < -1.5:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma_1d:
                desired_signal = POSITION_SIZE
            elif position_side < 0 and price_below_hma_1d:
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
        prev_fisher = current_fisher
    
    return signals