#!/usr/bin/env python3
"""
Experiment #601: 15m Mean Reversion with 4h Trend Filter + RSI Extremes

Hypothesis: After 532+ failures, the key insight is that 15m timeframe is ideal for
mean reversion strategies (captures noise/intraday swings) but needs HTF trend filter
to avoid counter-trend trades. This strategy combines:

1. 4h HMA(21) for trend bias (HTF via mtf_data helper - call ONCE before loop)
2. 15m RSI(7) for fast mean reversion signals (lower period = more trades)
3. 15m Bollinger Bands(20, 2.0) for entry confirmation at extremes
4. 15m ATR(14) for stoploss (2.0x ATR trailing)
5. Looser thresholds to ensure 10+ trades per symbol (Rule 9 - CRITICAL)

Why this should work on 15m:
- 15m captures intraday mean reversion (price oscillates around HTF trend)
- RSI(7) is faster than RSI(14) = more signals on 15m
- 4h HMA filter prevents trading against major trend
- BB confirmation ensures entries at statistical extremes
- Conservative sizing (0.25) controls drawdown during 2022 crash

Key differences from failed strategies:
- RSI(7) not RSI(14) = more trades on 15m timeframe
- BB touch not BB break = earlier entries, more signals
- No ADX filter = avoids filtering out valid mean reversion trades
- Looser RSI thresholds (25/75 not 20/80) = ensures trade count

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (get_htf_data called ONCE before loop)
Position sizing: 0.25 discrete (max 0.40)
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_meanrev_4h_hma_rsi7_bb_extremes_atr_v1"
timeframe = "15m"
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

def calculate_rsi(close, period=7):
    """Calculate RSI (Relative Strength Index) - faster period for 15m."""
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
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper.values, lower.values, sma.values

def calculate_percent_b(close, bb_upper, bb_lower):
    """Calculate %B position within Bollinger Bands."""
    bb_range = bb_upper - bb_lower
    percent_b = (close - bb_lower) / bb_range.replace(0, np.inf)
    return percent_b

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)  # Faster RSI for more 15m signals
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    percent_b = calculate_percent_b(close, bb_upper, bb_lower)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss (Rule 6)
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === 15M RSI EXTREMES (faster period = more signals) ===
        rsi_oversold = rsi_7[i] < 30  # Looser than 20 for more trades
        rsi_overbought = rsi_7[i] > 70  # Looser than 80 for more trades
        
        # === BOLLINGER BAND CONFIRMATION ===
        near_bb_lower = close[i] <= bb_lower[i] * 1.002  # At or below lower band
        near_bb_upper = close[i] >= bb_upper[i] * 0.998  # At or above upper band
        
        # === %B CONFIRMATION (extreme positions) ===
        percent_b_low = percent_b[i] < 0.1  # Bottom 10% of BB range
        percent_b_high = percent_b[i] > 0.9  # Top 10% of BB range
        
        # === ENTRY LOGIC - Mean Reversion with HTF Filter ===
        new_signal = 0.0
        
        # LONG: RSI oversold + BB lower + 4h trend not strongly bearish
        # Allow counter-trend longs when RSI is very extreme (RSI < 20)
        if rsi_oversold and (near_bb_lower or percent_b_low):
            if bull_bias:
                # With trend: standard entry
                new_signal = SIZE
            elif rsi_7[i] < 20:
                # Very oversold: allow counter-trend (mean reversion)
                new_signal = SIZE
            elif bear_bias and rsi_7[i] < 25:
                # Mildly oversold in bear: smaller position
                new_signal = SIZE * 0.6
        
        # SHORT: RSI overbought + BB upper + 4h trend not strongly bullish
        # Allow counter-trend shorts when RSI is very extreme (RSI > 80)
        if rsi_overbought and (near_bb_upper or percent_b_high):
            if bear_bias:
                # With trend: standard entry
                new_signal = -SIZE
            elif rsi_7[i] > 80:
                # Very overbought: allow counter-trend (mean reversion)
                new_signal = -SIZE
            elif bull_bias and rsi_7[i] > 75:
                # Mildly overbought in bull: smaller position
                new_signal = -SIZE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === MEAN REVERSION EXIT (RSI crosses back to neutral) ===
        mean_reversion_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_7[i] > 55:
                # Long exit when RSI recovers to neutral
                mean_reversion_exit = True
            if position_side < 0 and rsi_7[i] < 45:
                # Short exit when RSI recovers to neutral
                mean_reversion_exit = True
        
        # Apply stoploss or mean reversion exit
        if stoploss_triggered or mean_reversion_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals