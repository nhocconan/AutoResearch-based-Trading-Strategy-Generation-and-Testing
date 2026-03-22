#!/usr/bin/env python3
"""
Experiment #383: 12h Bollinger Mean Reversion + 1d/1w HMA Trend Bias + Vol Filter

Hypothesis: After 382 experiments, the key insight is that 12h timeframe needs
MEAN REVERSION logic (not pure trend following) with HTF trend BIAS (not filter).
Trend-following alone fails in 2022 crash and 2025 bear market. Pure mean-reversion
fails in strong trends. We combine them: mean-reversion entries WITH trend bias.

STRATEGY COMPONENTS:
1. 1d HMA(21) + 1w HMA(21) TREND BIAS (not filter):
   - Both bullish = long bias (take long mean-reversion, skip short)
   - Both bearish = short bias (take short mean-reversion, skip long)
   - Mixed = neutral (reduce position size by 50%)
   - This is softer than hard filter, allows trades in all regimes

2. BOLLINGER BAND MEAN REVERSION (20, 2.0):
   - Long when price < BB_lower + RSI(14) < 35
   - Short when price > BB_upper + RSI(14) > 65
   - BB mean-reversion works on 12h (proven in literature)

3. VOLATILITY SPIKE FILTER:
   - ATR(7)/ATR(30) > 1.8 = vol spike (panic/extreme = good MR opportunity)
   - Only enter when vol is elevated (captures "vol crush" after panic)
   - This avoids entering during quiet chop

4. ATR TRAILING STOP (2.5x):
   - Signal → 0 when price moves 2.5*ATR against position
   - Protects from trend runaway

5. POSITION SIZING: 0.30 discrete (conservative for 12h volatility)
   - Max 30% capital per position
   - Discrete levels minimize fee churn
   - Reduce to 0.15 when HTF signals mixed

Why this should work on 12h:
- 12h has fewer bars → fewer false signals than 15m/1h
- Mean-reversion works better on 12h than pure trend (less noise)
- HTF trend bias (1d/1w) provides direction without hard filtering
- Vol spike filter ensures we only enter at extremes (panic/euphoria)
- Should generate 20-40 trades/year per symbol (enough for stats)
- Works on BTC, ETH, SOL individually (not SOL-biased)

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 normal, 0.15 mixed bias
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_bb_meanrev_1d_1w_hma_vol_spike_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper.values, lower.values, sma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_NORMAL = 0.30
    SIZE_MIXED = 0.15
    
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] == 0:
            signals[i] = 0.0
            continue
        
        # === HTF TREND BIAS (1d + 1w HMA) ===
        bull_1d = close[i] > hma_1d_aligned[i]
        bear_1d = close[i] < hma_1d_aligned[i]
        bull_1w = close[i] > hma_1w_aligned[i]
        bear_1w = close[i] < hma_1w_aligned[i]
        
        # Determine bias strength
        if bull_1d and bull_1w:
            trend_bias = 1  # Strong long bias
        elif bear_1d and bear_1w:
            trend_bias = -1  # Strong short bias
        else:
            trend_bias = 0  # Mixed/neutral
        
        # === VOLATILITY SPIKE FILTER ===
        vol_ratio = atr_7[i] / (atr_30[i] + 1e-10)
        vol_spike = vol_ratio > 1.8  # Elevated volatility = MR opportunity
        
        # === BOLLINGER BAND MEAN REVERSION SIGNALS ===
        # Long: price below BB lower + RSI oversold
        bb_long = close[i] < bb_lower[i] and rsi[i] < 35
        
        # Short: price above BB upper + RSI overbought
        bb_short = close[i] > bb_upper[i] and rsi[i] > 65
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # Long entry: BB long signal + (long bias OR neutral) + vol spike
        if bb_long and vol_spike:
            if trend_bias >= 0:  # Long bias or neutral
                new_signal = SIZE_NORMAL if trend_bias == 1 else SIZE_MIXED
            # Skip long if strong short bias
        
        # Short entry: BB short signal + (short bias OR neutral) + vol spike
        elif bb_short and vol_spike:
            if trend_bias <= 0:  # Short bias or neutral
                new_signal = -SIZE_NORMAL if trend_bias == -1 else -SIZE_MIXED
            # Skip short if strong long bias
        
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
        
        # === MEAN REVERSION EXIT ===
        # Exit long when price returns to BB mid or RSI > 55
        if in_position and position_side > 0 and new_signal != 0.0:
            if close[i] >= bb_mid[i] or rsi[i] > 55:
                new_signal = 0.0
        
        # Exit short when price returns to BB mid or RSI < 45
        if in_position and position_side < 0 and new_signal != 0.0:
            if close[i] <= bb_mid[i] or rsi[i] < 45:
                new_signal = 0.0
        
        # === TREND BIAS REVERSAL EXIT ===
        # Exit long if trend bias flips strongly bearish
        if in_position and position_side > 0 and new_signal != 0.0:
            if trend_bias == -1:  # Strong short bias now
                new_signal = 0.0
        
        # Exit short if trend bias flips strongly bullish
        if in_position and position_side < 0 and new_signal != 0.0:
            if trend_bias == 1:  # Strong long bias now
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