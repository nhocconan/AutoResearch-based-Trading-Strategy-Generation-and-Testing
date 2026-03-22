#!/usr/bin/env python3
"""
Experiment #494: 30m Mean-Reversion with 4h HMA Trend Bias

Hypothesis: After 493 failed experiments, the pattern is clear - complex multi-filter
strategies fail because they're too restrictive. BTC/ETH spend 70%+ time in range
markets where mean-reversion dominates. This strategy uses:

1. 4H HMA(21) TREND BIAS (via mtf_data helper):
   - Price > 4h HMA = bull bias (favor long mean-reversion)
   - Price < 4h HMA = bear bias (favor short mean-reversion)

2. 30M RSI(7) FAST MEAN-REVERSION:
   - RSI(7) < 25 = oversold long entry
   - RSI(7) > 75 = overbought short entry
   - Faster than RSI(14) for 30m timeframe

3. BOLLINGER BAND POSITION CONFIRMATION:
   - Long: price < BB lower band (2.0 std)
   - Short: price > BB upper band (2.0 std)
   - Confirms extreme deviation from mean

4. ATR(14) STOPLOSS at 2.5x:
   - Tighter stop for 30m volatility
   - Signal → 0 when price moves 2.5*ATR against position

5. POSITION SIZING: 0.25 discrete
   - Conservative for 30m noise
   - Discrete levels minimize fee churn

Why this should work:
- Simpler = more trades (avoid 0-trade failure)
- Mean-reversion dominant in bear/range markets (2022, 2025)
- 4h HMA provides trend context without over-filtering
- RSI(7) + BB combo catches extremes reliably
- Should generate 50-100 trades/year per symbol

Timeframe: 30m (REQUIRED)
HTF: 4h via mtf_data helper
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_meanrev_4h_hma_rsi7_bb_atr_v1"
timeframe = "30m"
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
    """Calculate Relative Strength Index with fast period for 30m."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper.values, lower.values, sma.values

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
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 7)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        # LONG: RSI oversold + price below BB lower + bull bias preferred
        if rsi[i] < 25 and close[i] < bb_lower[i]:
            if bull_bias:
                new_signal = SIZE  # Strong long in bull bias
            else:
                new_signal = SIZE * 0.6  # Weaker long in bear bias
        
        # SHORT: RSI overbought + price above BB upper + bear bias preferred
        if rsi[i] > 75 and close[i] > bb_upper[i]:
            if bear_bias:
                new_signal = -SIZE  # Strong short in bear bias
            else:
                new_signal = -SIZE * 0.6  # Weaker short in bull bias
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if position_side != 0:
            stopped_out = False
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    stopped_out = True
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    stopped_out = True
            
            if stopped_out:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if position_side == 0:
                # New position
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: same side, update extremes below
        else:
            # No signal = exit position
            position_side = 0
            highest_close = 0.0
            lowest_close = 0.0
        
        # Update extremes for active positions (even if signal unchanged)
        if position_side > 0 and new_signal > 0:
            if close[i] > highest_close:
                highest_close = close[i]
        if position_side < 0 and new_signal < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
        
        signals[i] = new_signal
    
    return signals