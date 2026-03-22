#!/usr/bin/env python3
"""
Experiment #468: 1d Daily Mean Reversion with Weekly Trend Bias

Hypothesis: After 467 experiments, the key insight is that daily strategies need
SIMPLE entry conditions to generate sufficient trades. Complex ensembles with
multiple filters (#461, #464) create conflicting conditions = 0 trades or negative Sharpe.

This strategy uses:

1. WEEKLY HMA(21) TREND BIAS (via mtf_data helper):
   - Long bias when price > 1w HMA
   - Short bias when price < 1w HMA
   - BUT: allow counter-trend mean reversion with smaller size

2. DAILY RSI(14) MEAN REVERSION (primary signal):
   - RSI < 25 = oversold (long)
   - RSI > 75 = overbought (short)
   - LOOSE thresholds to ensure trades on all symbols

3. BOLLINGER BAND(20, 2.0) CONFIRMATION:
   - Price must touch/ breach BB for entry
   - Confirms extreme move, not just RSI

4. ATR(14) TRAILING STOP at 2.5x:
   - Signal → 0 when price moves 2.5*ATR against position
   - Critical for crash protection

5. POSITION SIZING: 0.25-0.30 discrete
   - 0.30 for trend-aligned entries
   - 0.20 for counter-trend mean reversion
   - Discrete levels minimize fee churn

Why this should work on 1d:
- Simple RSI+BB combo generates reliable mean reversion signals
- Weekly HMA provides bias without blocking counter-trend trades
- Looser thresholds than failed experiments ensure >10 trades/year
- Should work on BTC/ETH/SOL individually (not SOL-biased)
- Daily timeframe captures multi-day swings perfectly

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_rsi_bb_weekly_hma_meanrev_atr_v1"
timeframe = "1d"
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

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + (std * std_mult)
    lower = sma - (std * std_mult)
    return upper.values, lower.values, sma.values

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    return close_s.rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    sma_50 = calculate_sma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_TREND = 0.30  # 30% when aligned with weekly trend
    SIZE_COUNTER = 0.20  # 20% for counter-trend mean reversion
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_50[i]):
            signals[i] = 0.0
            continue
        
        # === WEEKLY HMA TREND BIAS ===
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === RSI MEAN REVERSION SIGNALS ===
        # Loose thresholds to ensure trades on all symbols
        rsi_oversold = rsi[i] < 25
        rsi_overbought = rsi[i] > 75
        
        # === BOLLINGER BAND CONFIRMATION ===
        # Price must touch or breach BB extremes
        bb_touch_lower = low[i] <= bb_lower[i]
        bb_touch_upper = high[i] >= bb_upper[i]
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG SIGNALS
        if rsi_oversold and bb_touch_lower:
            if bull_trend_1w:
                # Trend-aligned mean reversion (higher conviction)
                new_signal = SIZE_TREND
            else:
                # Counter-trend mean reversion (lower conviction)
                new_signal = SIZE_COUNTER
        
        # SHORT SIGNALS
        if rsi_overbought and bb_touch_upper:
            if bear_trend_1w:
                # Trend-aligned mean reversion (higher conviction)
                new_signal = -SIZE_TREND
            else:
                # Counter-trend mean reversion (lower conviction)
                new_signal = -SIZE_COUNTER
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if weekly trend flips strongly against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_1w:
                # Long in bear trend - reduce or exit
                if close[i] < sma_50[i]:
                    new_signal = 0.0
            if position_side < 0 and bull_trend_1w:
                # Short in bull trend - reduce or exit
                if close[i] > sma_50[i]:
                    new_signal = 0.0
        
        # === RSI EXIT (mean reversion complete) ===
        # Exit when RSI returns to neutral
        if in_position and new_signal != 0.0:
            if position_side > 0 and rsi[i] > 55:
                # Long position, RSI recovered - take profit
                new_signal = 0.0
            if position_side < 0 and rsi[i] < 45:
                # Short position, RSI recovered - take profit
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals