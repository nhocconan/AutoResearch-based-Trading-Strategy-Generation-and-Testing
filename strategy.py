#!/usr/bin/env python3
"""
Experiment #423: 1h Bollinger Squeeze + 4h HMA Trend + Volatility Expansion

Hypothesis: After 422 failed experiments, the pattern is clear:
1. Pure trend-following fails on BTC/ETH (2022 crash + 2025 bear)
2. Pure mean-reversion fails in strong trends  
3. Need VOLATILITY-BASED entry with TREND BIAS

This strategy uses:
1. BOLLINGER BAND SQUEEZE on 1h: BW < 25th percentile = low vol (coiling)
2. VOLATILITY EXPANSION: BBW expanding from squeeze = breakout imminent
3. 4h HMA(21) TREND BIAS: Long only when price > 4h HMA, short when <
4. RSI(14) FILTER: Avoid entries at extremes (RSI 40-60 sweet spot for breakouts)
5. ATR(14) STOPLOSS: 2.0x ATR trailing stop

Why this should work:
- Squeeze captures "calm before storm" - works in both bull/bear
- 4h HMA provides trend bias without being too strict
- RSI filter avoids chasing extended moves
- Conservative sizing (0.25) limits drawdown in 2022-style crashes
- Works on 1h = more trades than 4h/12h strategies (meets trade count requirement)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper
Position sizing: 0.25 discrete
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_bb_squeeze_4h_hma_vol_expansion_atr_v1"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma * 100
    return upper.values, lower.values, bandwidth.values, sma.values

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

def calculate_bbw_percentile(bb_width, lookback=100):
    """Calculate rolling percentile of BB Width for squeeze detection."""
    n = len(bb_width)
    percentile = np.full(n, np.nan)
    
    for i in range(lookback, n):
        window = bb_width[i-lookback:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            percentile[i] = np.sum(valid <= bb_width[i]) / len(valid) * 100
    
    return percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_width, bb_sma = calculate_bollinger(close, 20, 2.0)
    bbw_percentile = calculate_bbw_percentile(bb_width, 100)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_width[i]) or np.isnan(bbw_percentile[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === SQUEEZE DETECTION ===
        squeeze = bbw_percentile[i] < 25  # BW in bottom 25% = compression
        
        # === VOLATILITY EXPANSION ===
        vol_expansion = False
        if i > 0 and not np.isnan(bb_width[i-1]) and bb_width[i-1] > 0:
            vol_expansion = bb_width[i] > bb_width[i-1] * 1.03  # 3% expansion
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === RSI FILTER ===
        # For breakouts: avoid extreme RSI (chasing) but allow momentum
        rsi_ok_long = 40 < rsi[i] < 70  # Not oversold, not extremely overbought
        rsi_ok_short = 30 < rsi[i] < 60  # Not overbought, not extremely oversold
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # Long: squeeze + expansion + bull trend + RSI ok
        if squeeze and vol_expansion and bull_trend_4h and rsi_ok_long:
            new_signal = SIZE
        
        # Short: squeeze + expansion + bear trend + RSI ok
        elif squeeze and vol_expansion and bear_trend_4h and rsi_ok_short:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
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
                in_position = False
                position_side = 0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals