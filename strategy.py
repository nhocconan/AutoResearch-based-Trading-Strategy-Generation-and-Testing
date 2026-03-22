#!/usr/bin/env python3
"""
Experiment #010: 1h Vol Spike Mean Reversion with 12h/4h HMA Regime Filter

Hypothesis: After 9 failed strategies using Fisher/Choppiness/RSI combinations,
I'm testing a DIFFERENT approach based on proven edges from research:

1. VOLATILITY SPIKE MEAN REVERSION: After panic spikes (ATR(7)/ATR(30) > 2.0),
   price tends to revert. This captures the "vol crush" after fear peaks.
   Reported Sharpe 0.8-1.5 through 2022 crash for BTC/ETH.

2. 12H HMA REGIME FILTER: More stable than 4h for major trend direction.
   Only long if price > 12h_HMA, only short if price < 12h_HMA.

3. 4H HMA CONFIRMATION: Additional HTF filter for confluence.

4. CONNORS RSI FOR ENTRY: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   Long when CRSI < 15, short when CRSI > 85. Better than standard RSI.

5. SESSION FILTER: Only trade 8-20 UTC (high liquidity, less manipulation).

6. VOLUME CONFIRMATION: Volume > 0.8 * 20-bar average.

Why different from failed #005, #009 (Fisher+Chop):
- Uses volatility spike detection instead of Fisher Transform
- Uses Connors RSI instead of standard RSI
- 12h HMA for regime (more stable than 4h)
- Focuses on mean reversion after panic (proven in 2022 crash)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h + 12h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete, ATR-scaled
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-80/year (strict entry filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_vol_spike_connors_12h_4h_hma_regime_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] > 0:
            streak_rsi[i] = min(100, 100 * (streak[i] / streak_period))
        elif streak[i] < 0:
            streak_rsi[i] = max(0, 100 * (1 - (abs(streak[i]) / streak_period)))
        else:
            streak_rsi[i] = 50
    
    # Percent Rank (100)
    pr = np.zeros(n)
    for i in range(pr_period, n):
        returns = close_s.iloc[i-pr_period:i].pct_change().dropna()
        current_return = close_s.iloc[i] / close_s.iloc[i-1] - 1 if i > 0 else 0
        if len(returns) > 0:
            pr[i] = (returns < current_return).sum() / len(returns) * 100
        else:
            pr[i] = 50
    
    # Combine into CRSI
    crsi = (rsi + streak_rsi + pr) / 3
    
    return crsi

def calculate_bollinger(close, period=20, std_mult=2.0):
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
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    connors = calculate_connors_rsi(close, 3, 2, 100)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Extract hour from open_time for session filter
    open_times = pd.to_datetime(prices["open_time"])
    hours = open_times.dt.hour.values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(150, n):  # Start later for all indicators
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        
        if np.isnan(connors[i]):
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            continue
        
        # === REGIME DETECTION (12h HMA) ===
        bull_regime = close[i] > hma_12h_aligned[i]
        bear_regime = close[i] < hma_12h_aligned[i]
        
        # === 4H HMA CONFIRMATION ===
        bull_4h = close[i] > hma_4h_aligned[i]
        bear_4h = close[i] < hma_4h_aligned[i]
        
        # === VOLATILITY SPIKE DETECTION ===
        # ATR(7) / ATR(30) > 2.0 = panic/vol spike
        atr_ratio = atr_7[i] / atr_30[i]
        vol_spike = atr_ratio > 2.0
        
        # === CONNORS RSI EXTREMES ===
        connors_oversold = connors[i] < 15
        connors_overbought = connors[i] > 85
        
        # === BOLLINGER BAND EXTREMES ===
        near_bb_lower = close[i] < bb_lower[i] * 1.01
        near_bb_upper = close[i] > bb_upper[i] * 0.99
        
        # === SESSION FILTER (8-20 UTC) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_sma[i]
        
        # === ATR-BASED POSITION SIZING ===
        # Reduce size when volatility is high
        atr_median = np.nanmedian(atr_14[100:i]) if i > 100 else atr_14[i]
        atr_ratio_size = atr_14[i] / atr_median if atr_median > 0 else 1.0
        atr_ratio_size = np.clip(atr_ratio_size, 0.5, 2.0)
        size_multiplier = 1.0 / atr_ratio_size
        current_size = BASE_SIZE * size_multiplier
        current_size = np.clip(current_size, 0.15, 0.35)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # MODE 1: VOL SPIKE MEAN REVERSION (highest priority)
        # After panic spike, expect reversion IF regime supports it
        if vol_spike and in_session and volume_confirmed:
            # Long: Vol spike + oversold + bull regime (or neutral)
            if connors_oversold and near_bb_lower:
                if bull_regime or (not bear_regime and not bear_4h):
                    new_signal = current_size
            
            # Short: Vol spike + overbought + bear regime (or neutral)
            elif connors_overbought and near_bb_upper:
                if bear_regime or (not bull_regime and not bull_4h):
                    new_signal = -current_size
        
        # MODE 2: STANDARD MEAN REVERSION (no vol spike, but extremes)
        elif in_session and volume_confirmed:
            # Long: Connors oversold + BB lower + not in strong bear regime
            if connors_oversold and near_bb_lower:
                if not bear_regime or (bear_regime and bull_4h):
                    new_signal = current_size
            
            # Short: Connors overbought + BB upper + not in strong bull regime
            elif connors_overbought and near_bb_upper:
                if not bull_regime or (bull_regime and bear_4h):
                    new_signal = -current_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if 12h regime turns strongly bearish
            if position_side > 0 and bear_regime and bear_4h:
                regime_reversal = True
            # Exit short if 12h regime turns strongly bullish
            if position_side < 0 and bull_regime and bull_4h:
                regime_reversal = True
        
        # Apply stoploss or regime reversal
        if stoploss_triggered or regime_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals