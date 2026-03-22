#!/usr/bin/env python3
"""
Experiment #261: 1h Volatility Spike Mean Reversion with 4h/12h Trend Filter

Hypothesis: Volatility spike mean reversion works across all market regimes (bull/bear/range).
When ATR(7)/ATR(30) > 2.0, volatility is elevated and likely to compress. 
Entering at Bollinger Band extremes AFTER vol spike captures the "vol crush" reversal.
4h/12h HMA provides directional bias to avoid counter-trend trades.

Why this might work:
- Volatility mean reversion is one of the most robust edges in crypto
- Works in 2022 crash (high vol) and 2025 bear (choppy high vol)
- 1h timeframe captures intraday vol spikes that 4h misses
- 4h/12h HMA filter prevents counter-trend mean reversion (major killer)
- Fewer but higher quality trades = less fee drag
- Based on research: "VOL SPIKE REVERSION: ATR(7)/ATR(30) > 2.0 + price < BB(20,2.5) → long"

Key differences from failed experiments:
- #255 (1h chop regime): Used choppy index, failed. This uses ATR ratio instead.
- #259 (15m RSI meanrev): Too fast timeframe, got whipsawed. 1h is slower.
- #254 (30m RSI pullback): RSI alone fails. This uses BB extremes + vol spike.
- Simpler entry logic = more trades generated (avoid 0-trade failure)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h and 12h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 base, 0.35 with HTF conviction
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_vol_spike_meanrev_4h_12h_hma_bb_atr_v1"
timeframe = "1h"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger_bands(close, period=20, std_mult=2.5):
    """Calculate Bollinger Bands with configurable std multiplier."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + (std_mult * std)
    lower = sma - (std_mult * std)
    return upper.values, lower.values, sma.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_14 = calculate_atr(high, low, close, 14)
    atr_30 = calculate_atr(high, low, close, 30)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.5)
    ema_50 = calculate_ema(close, 50)
    
    # Volatility spike ratio: ATR(7) / ATR(30)
    vol_ratio = atr_7 / (atr_30 + 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HIGH = 0.35
    SIZE_HALF = 0.125
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME TREND BIAS ===
        # 4h HMA = intermediate trend
        # 12h HMA = longer trend (stronger conviction)
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        bull_trend_12h = close[i] > hma_12h_aligned[i]
        bear_trend_12h = close[i] < hma_12h_aligned[i]
        
        # === VOLATILITY SPIKE DETECTION ===
        # Vol spike when ATR(7) > 2.0 * ATR(30)
        vol_spike = vol_ratio[i] > 2.0
        
        # === BOLLINGER BAND POSITION ===
        # Price at lower band = oversold (long candidate)
        # Price at upper band = overbought (short candidate)
        at_lower_band = close[i] <= bb_lower[i] * 1.002  # 0.2% tolerance
        at_upper_band = close[i] >= bb_upper[i] * 0.998  # 0.2% tolerance
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        # LONG ENTRY: Vol spike + price at lower BB + trend bias
        # Need: vol_spike AND at_lower_band AND (bull_trend_4h OR bull_trend_12h)
        # Relaxed: allow long even in bear trend if vol spike is extreme (>2.5)
        extreme_vol_spike = vol_ratio[i] > 2.5
        
        if vol_spike and at_lower_band:
            if bull_trend_4h or bull_trend_12h:
                # Trend-aligned mean reversion
                new_signal = SIZE_BASE
                if bull_trend_12h:
                    new_signal = SIZE_HIGH  # Higher conviction
            elif extreme_vol_spike:
                # Extreme vol spike overrides trend (panic bottom)
                new_signal = SIZE_BASE
        
        # SHORT ENTRY: Vol spike + price at upper BB + trend bias
        if vol_spike and at_upper_band:
            if bear_trend_4h or bear_trend_12h:
                # Trend-aligned mean reversion
                new_signal = -SIZE_BASE
                if bear_trend_12h:
                    new_signal = -SIZE_HIGH  # Higher conviction
            elif extreme_vol_spike:
                # Extreme vol spike overrides trend (panic top)
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TAKE PROFIT: Reduce to half at 2R profit ===
        if in_position and new_signal != 0.0 and position_side != 0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * entry_atr:
                    new_signal = SIZE_HALF  # Take partial profit
            if position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * entry_atr:
                    new_signal = -SIZE_HALF  # Take partial profit
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly adjusted size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals