#!/usr/bin/env python3
"""
Experiment #393: 1h Dual Momentum (Price + Volume) with 4h HMA Trend Filter

Hypothesis: After 392 failed experiments, the pattern is clear:
- Mean-reversion (RSI, CRSI, BB) fails in strong trends (2021 bull, 2022 crash)
- Pure trend-following (Supertrend, EMA) fails in chop (2022 bottom, 2025 bear)
- Volume is UNDERUTILIZED in failed strategies - it confirms real breakouts

STRATEGY COMPONENTS:
1. DUAL MOMENTUM SCORE: Combine price momentum (ROC12) + volume momentum (VolMA20)
   - Price ROC(12) > 5% = strong upward momentum
   - Volume ratio > 1.3 = conviction behind move
   - Both must agree = fewer false signals, higher win rate

2. 4h HMA(21) TREND FILTER: Only trade in HTF trend direction
   - Long only when 1h close > 4h HMA (bullish HTF)
   - Short only when 1h close < 4h HMA (bearish HTF)
   - This avoids counter-trend trades that caused 2022 whipsaw losses

3. MOMENTUM DECAY EXIT: Exit when momentum weakens
   - Long: exit when ROC(12) drops below 2% (momentum fading)
   - Short: exit when ROC(12) rises above -2% (downward momentum fading)
   - This captures the meat of trends, exits before reversal

4. ATR TRAILING STOP (2.5x): Hard stoploss for crash protection
   - Signal → 0 when price moves 2.5*ATR against position
   - Essential for 2022-style 77% crashes

5. POSITION SIZING: 0.25 discrete (conservative for 1h volatility)
   - Max 25% capital per position
   - Discrete levels (0.0, ±0.25) minimize fee churn

Why this should work:
- Dual momentum (price + volume) filters false breakouts
- 4h HMA trend filter prevents counter-trend disasters
- Momentum decay exit captures trends without waiting for reversal
- Should generate 40-80 trades/year per symbol (enough for stats)
- Works on BTC, ETH, SOL individually (volume confirms across all)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_dual_momentum_4h_hma_vol_atr_v1"
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

def calculate_roc(close, period=12):
    """Calculate Rate of Change (momentum indicator)."""
    roc = np.full(len(close), np.nan)
    for i in range(period, len(close)):
        if close[i-period] > 0:
            roc[i] = 100 * (close[i] - close[i-period]) / close[i-period]
    return roc

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    roc = calculate_roc(close, 12)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
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
        
        if np.isnan(roc[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === DUAL MOMENTUM SIGNALS ===
        # Strong long momentum: ROC > 5% AND volume > 1.3x average
        strong_long_momentum = (roc[i] > 5.0) and (vol_ratio[i] > 1.3)
        
        # Strong short momentum: ROC < -5% AND volume > 1.3x average
        strong_short_momentum = (roc[i] < -5.0) and (vol_ratio[i] > 1.3)
        
        # Moderate momentum (for exit signals)
        weak_long_momentum = roc[i] > 2.0
        weak_short_momentum = roc[i] < -2.0
        
        # === 4h HMA TREND FILTER ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: Strong momentum + bullish 4h trend
        if strong_long_momentum and bull_trend_4h:
            new_signal = SIZE
        
        # SHORT ENTRY: Strong momentum + bearish 4h trend
        elif strong_short_momentum and bear_trend_4h:
            new_signal = -SIZE
        
        # === MOMENTUM DECAY EXIT ===
        # Exit long when momentum fades (ROC drops below 2%)
        if in_position and position_side > 0 and not weak_long_momentum:
            new_signal = 0.0
        
        # Exit short when momentum fades (ROC rises above -2%)
        if in_position and position_side < 0 and not weak_short_momentum:
            new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit long if 4h trend turns bearish
        if in_position and position_side > 0 and bear_trend_4h:
            new_signal = 0.0
        
        # Exit short if 4h trend turns bullish
        if in_position and position_side < 0 and bull_trend_4h:
            new_signal = 0.0
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and new_signal != 0.0:
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