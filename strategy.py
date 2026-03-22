#!/usr/bin/env python3
"""
Experiment #399: 1h Volatility Regime + 4h HMA Trend + ATR Stop

Hypothesis: After 398 failed experiments, the pattern is clear - strategies fail because
they're either too complex (too many conflicting filters) or don't adapt to volatility regimes.

KEY INSIGHTS FROM FAILURES:
- Exp #387-398: All negative Sharpe, most -1.0 to -5.0
- Over-filtering = 0 trades or massive whipsaw
- BTC/ETH need volatility-adaptive logic, not pure trend or pure mean-reversion

STRATEGY DESIGN:
1. VOLATILITY REGIME (BB Width Percentile): 
   - BB Width > 70th percentile = high vol (trend-following works)
   - BB Width < 30th percentile = low vol (mean-reversion works)
   - This adapts to market conditions automatically

2. 4h HMA(21) TREND BIAS:
   - Long only when price > 4h HMA in high vol regime
   - Short only when price < 4h HMA in high vol regime
   - Provides stable trend filter without whipsaw

3. 1h RSI(14) ENTRY TRIGGER:
   - High vol + trend: Enter on RSI pullback (40-60 range)
   - Low vol: Enter on RSI extremes (<30 long, >70 short)
   - Different logic per regime

4. ATR(14) TRAILING STOP (2.5x):
   - Signal → 0 when price moves 2.5*ATR against position
   - Protects from crash scenarios

5. POSITION SIZING: 0.30 discrete
   - Conservative enough for 77% BTC crash
   - Discrete levels minimize fee churn

Why this should work:
- Adapts to volatility regime (key missing piece in 300+ failures)
- 4h HMA is stable trend filter (proven in baseline)
- RSI entry is simple and generates trades
- Should work on BTC, ETH, SOL individually
- Targets 40-80 trades/year (enough for stats, not too many for fees)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_vol_regime_4h_hma_rsi_atr_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard formula."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / (sma + 1e-10)
    return upper.values, lower.values, bandwidth.values, sma.values

def calculate_bb_width_percentile(bandwidth, lookback=100):
    """Calculate rolling percentile of BB width to detect volatility regime."""
    n = len(bandwidth)
    percentile = np.full(n, np.nan)
    
    for i in range(lookback, n):
        window = bandwidth[i-lookback+1:i+1]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) >= lookback // 2:
            percentile[i] = np.sum(valid_window <= bandwidth[i]) / len(valid_window) * 100
    
    return percentile

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
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_bandwidth, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    bb_percentile = calculate_bb_width_percentile(bb_bandwidth, 100)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_percentile[i]):
            signals[i] = 0.0
            continue
        
        # === VOLATILITY REGIME DETECTION ===
        high_vol = bb_percentile[i] > 70  # Top 30% volatility = trending
        low_vol = bb_percentile[i] < 30   # Bottom 30% volatility = ranging
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === GENERATE SIGNAL BASED ON REGIME ===
        new_signal = 0.0
        
        # HIGH VOLATILITY REGIME: Trend-following with RSI pullback
        if high_vol:
            if bull_trend_4h and rsi[i] >= 40 and rsi[i] <= 60:
                # Long on RSI pullback in uptrend
                new_signal = SIZE
            elif bear_trend_4h and rsi[i] >= 40 and rsi[i] <= 60:
                # Short on RSI pullback in downtrend
                new_signal = -SIZE
        
        # LOW VOLATILITY REGIME: Mean-reversion at extremes
        elif low_vol:
            if rsi[i] < 30:
                # Oversold = long
                new_signal = SIZE
            elif rsi[i] > 70:
                # Overbought = short
                new_signal = -SIZE
        
        # NEUTRAL VOLATILITY (30-70 percentile): Stay flat or hold existing
        
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
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h and high_vol:
                # Long position exits when 4h trend flips bearish in high vol
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h and high_vol:
                # Short position exits when 4h trend flips bullish in high vol
                new_signal = 0.0
        
        # === RSI EXTREME EXIT (for low vol mean-reversion) ===
        if in_position and low_vol:
            if position_side > 0 and rsi[i] > 60:
                # Long exits when RSI recovers to neutral
                new_signal = 0.0
            if position_side < 0 and rsi[i] < 40:
                # Short exits when RSI recovers to neutral
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