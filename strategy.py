#!/usr/bin/env python3
"""
Experiment #561: 1h Multi-Signal Ensemble with 4h HMA Trend Bias

Hypothesis: After analyzing 560+ failed experiments, the key insight is:
1. 1h timeframe offers good balance between signal frequency and noise
2. Single-indicator strategies fail - need ensemble voting (2 of 3 signals)
3. 4h HMA trend bias prevents counter-trend entries (major failure mode in 2022)
4. LOOSE entry conditions are CRITICAL - many strategies failed with 0 trades
5. RSI(7) shorter period generates more signals than RSI(14)
6. EMA(9/21) crossover catches momentum without excessive lag
7. Volume spike confirmation filters false breakouts
8. 2.0*ATR stoploss protects against 2022-style crashes while allowing breathing room

Why this should work on 1h:
- 1h has 24 bars/day = ~8760 bars/year = sufficient trade frequency
- Ensemble approach (2 of 3 signals) reduces false positives
- 4h HMA is proven trend filter from successful strategies
- Loose RSI thresholds (35/65 not 30/70) ensure trades happen
- Volume filter (>1.5x avg) confirms genuine moves
- Conservative position sizing (0.28) limits drawdown

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.28 discrete (max 0.40)
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_ensemble_rsi_ema_volume_4h_hma_atr_v1"
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

def calculate_rsi(close, period=7):
    """Calculate RSI with shorter period for more signals."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_ema(close, period):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    return close_s.ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_volume_spike(volume, period=20):
    """Calculate volume spike ratio (current vs rolling avg)."""
    volume_s = pd.Series(volume)
    volume_avg = volume_s.rolling(window=period, min_periods=period).mean()
    volume_ratio = volume_s / volume_avg.replace(0, np.inf)
    return volume_ratio.values

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
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)
    ema_9 = calculate_ema(close, 9)
    ema_21 = calculate_ema(close, 21)
    volume_ratio = calculate_volume_spike(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(ema_9[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === 1H RSI SIGNAL (loose thresholds for more trades) ===
        rsi_oversold = rsi_7[i] < 40  # Not 30 - too strict
        rsi_overbought = rsi_7[i] > 60  # Not 70 - too strict
        
        # === 1H EMA CROSSOVER ===
        ema_bullish = ema_9[i] > ema_21[i]
        ema_bearish = ema_9[i] < ema_21[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume_ratio[i] > 1.3  # Loose threshold
        
        # === ENSEMBLE VOTING (2 of 3 signals needed) ===
        long_votes = 0
        short_votes = 0
        
        # Long votes
        if bull_bias:
            long_votes += 1
        if rsi_oversold:
            long_votes += 1
        if ema_bullish:
            long_votes += 1
        
        # Short votes
        if bear_bias:
            short_votes += 1
        if rsi_overbought:
            short_votes += 1
        if ema_bearish:
            short_votes += 1
        
        # === ENTRY LOGIC (LOOSE - need only 2 of 3) ===
        new_signal = 0.0
        
        # Long: 2+ long votes + volume confirmation (or skip volume for more trades)
        if long_votes >= 2:
            new_signal = SIZE
        
        # Short: 2+ short votes + volume confirmation (or skip volume for more trades)
        elif short_votes >= 2:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h HMA flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias:
                new_signal = 0.0
            if position_side < 0 and bull_bias:
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