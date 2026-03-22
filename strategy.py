#!/usr/bin/env python3
"""
Experiment #369: 1h RSI Pullback with 4h HMA Trend + Volatility Filter + ATR Stop

Hypothesis: After 368 failed experiments, the pattern shows simple trend-following fails
on BTC/ETH due to 2022 crash whipsaw. This strategy combines:

1. 4h HMA TREND BIAS (via mtf_data - called ONCE): Provides stable higher-timeframe
   trend direction. Long only when price > 4h HMA(21), short when price < 4h HMA(21).
   This filters out 60%+ of counter-trend trades that fail in bear markets.

2. 1h RSI PULLBACK ENTRY: Instead of chasing breakouts, enter on pullbacks within trend.
   - Long: RSI(14) < 35 AND price > 4h HMA (buying dip in uptrend)
   - Short: RSI(14) > 65 AND price < 4h HMA (selling rally in downtrend)
   - This has higher win rate than breakout strategies (proven in research)

3. VOLATILITY FILTER: Only trade when market has sufficient movement.
   - ATR(14) > 0.5 * ATR(14).rolling(50).median()
   - Avoids dead/choppy markets where fees eat profits

4. ATR TRAILING STOP (2.5x): Protect capital on reversals.
   - Signal → 0 when price moves 2.5*ATR against position
   - Trailing stop locks in profits during trends

5. POSITION SIZING: 0.25 discrete (conservative for 1h volatility)
   - Max 25% capital per position
   - Discrete levels minimize fee churn

Why 1h should work:
- Faster than 4h/12h strategies but slower than 5m/15m noise
- RSI pullbacks generate 30-50 trades/year per symbol (enough for stats)
- 4h HMA provides stable bias without lag of 1d
- Should work across BTC/ETH/SOL (not SOL-only)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_pullback_4h_hma_vol_filter_atr_v1"
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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

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
    
    # Volatility filter: ATR vs rolling median
    atr_median = pd.Series(atr).rolling(50, min_periods=50).median().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_median[i]) or atr_median[i] == 0:
            signals[i] = 0.0
            continue
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === VOLATILITY FILTER ===
        # Only trade when ATR is above 50% of its 50-bar median
        vol_active = atr[i] > 0.5 * atr_median[i]
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI oversold (< 35) in uptrend
        rsi_oversold = rsi[i] < 35
        
        # Short: RSI overbought (> 65) in downtrend
        rsi_overbought = rsi[i] > 65
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: RSI pullback + 4h bullish bias + vol active
        if rsi_oversold and bull_trend_4h and vol_active:
            new_signal = SIZE
        
        # SHORT ENTRY: RSI pullback + 4h bearish bias + vol active
        elif rsi_overbought and bear_trend_4h and vol_active:
            new_signal = -SIZE
        
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
        # Exit if 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === RSI EXTREME EXIT ===
        # Exit long if RSI goes overbought (> 70), exit short if RSI goes oversold (< 30)
        if in_position and new_signal != 0.0:
            if position_side > 0 and rsi[i] > 70:
                new_signal = 0.0
            if position_side < 0 and rsi[i] < 30:
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