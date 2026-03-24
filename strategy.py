#!/usr/bin/env python3
"""
Experiment #027: 6h Primary + 1d HTF — CRSI Mean Reversion + HMA Trend Filter

Hypothesis: After 26 failed experiments, 6h timeframe needs a fundamentally different approach.
Key insight from #026: CRSI + Choppiness + HMA on 1d achieved Sharpe=0.167 (current best).
This strategy applies that proven pattern to 6h with these modifications:
- Connors RSI (CRSI) for mean reversion entries (75% win rate in literature)
- 1d HMA(50) for major trend bias (filters counter-trend trades)
- Volume spike confirmation (1.5x avg) to ensure real moves
- Looser CRSI thresholds (15/85 instead of 10/90) to ensure >=30 trades on train
- Position size 0.28 (conservative for 6h volatility)
- Stoploss: 2.5x ATR trailing

Why this should work on 6h:
- 6h is middle ground: fewer false signals than 1h, more trades than 12h
- CRSI captures short-term oversold/overbought better than standard RSI
- 1d HMA provides stable trend filter without being too restrictive
- Volume filter reduces fake breakouts common in crypto

Target: Sharpe>0.167, DD>-40%, trades>=30 train, trades>=3 test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_crsi_hma_volume_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    Formula: (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(close, 3): Standard RSI on price with 3-period
    RSI(streak, 2): RSI on consecutive up/down streaks with 2-period
    PercentRank(100): Percentile rank of today's close vs last 100 closes
    
    CRSI < 10 = extremely oversold (long signal)
    CRSI > 90 = extremely overbought (short signal)
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    for i in range(rank_period, n):
        # RSI(close, 3)
        delta = np.diff(close[:i+1])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        
        if len(gain) >= rsi_period:
            avg_gain = np.mean(gain[-rsi_period:])
            avg_loss = np.mean(loss[-rsi_period:])
            if avg_loss < 1e-10:
                rsi_close = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_close = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi_close = 50.0
        
        # RSI(streak, 2) - calculate streak sequence
        streaks = np.zeros(i + 1)
        streaks[0] = 0
        for j in range(1, i + 1):
            if close[j] > close[j-1]:
                streaks[j] = streaks[j-1] + 1 if streaks[j-1] >= 0 else 1
            elif close[j] < close[j-1]:
                streaks[j] = streaks[j-1] - 1 if streaks[j-1] <= 0 else -1
            else:
                streaks[j] = streaks[j-1]
        
        # Convert streaks to RSI input (positive = up streak, negative = down)
        streak_delta = np.diff(streaks)
        streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
        streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
        
        if len(streak_gain) >= streak_period:
            avg_streak_gain = np.mean(streak_gain[-streak_period:]) if len(streak_gain) >= streak_period else 0.0
            avg_streak_loss = np.mean(streak_loss[-streak_period:]) if len(streak_loss) >= streak_period else 0.0
            if avg_streak_loss < 1e-10:
                rsi_streak = 100.0
            else:
                rs_streak = avg_streak_gain / avg_streak_loss
                rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
        else:
            rsi_streak = 50.0
        
        # PercentRank(100): percentile of current close vs last 100 closes
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window[:-1] < close[i]) / (rank_period - 1)
        percent_rank = rank * 100.0
        
        # CRSI = average of three components
        crsi[i] = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_spike(volume, period=20, threshold=1.5):
    """Detect volume spikes (volume > threshold * avg_volume)"""
    n = len(volume)
    if n < period:
        return np.full(n, False)
    
    avg_vol = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    spike = volume > (threshold * avg_vol)
    return spike

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr = calculate_atr(high, low, close, period=14)
    vol_spike = calculate_volume_spike(volume, period=20, threshold=1.5)
    
    # 6h HMA for local trend
    hma_6h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (conservative for 6h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(hma_6h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === CRSI MEAN REVERSION SIGNALS ===
        # LOOSE thresholds to ensure trades generate (15/85 instead of 10/90)
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === VOLUME CONFIRMATION ===
        # Volume spike confirms the move is real
        vol_confirm = vol_spike[i]
        
        # === 6h HMA LOCAL TREND ===
        hma_bull = close[i] > hma_6h[i]
        hma_bear = close[i] < hma_6h[i]
        
        # === DESIRED SIGNAL (CRSI Mean Reversion + HTF Filter) ===
        desired_signal = 0.0
        
        # LONG: CRSI oversold + HTF not strongly bear (or HTF bull)
        # Allow longs even if HTF neutral, just not strongly bear
        if crsi_oversold:
            if htf_bull:
                # Strong signal: HTF bull + CRSI oversold
                desired_signal = SIZE
            elif not htf_bear:
                # Moderate signal: HTF neutral + CRSI oversold + volume
                if vol_confirm or hma_bull:
                    desired_signal = SIZE * 0.7
            elif htf_bear and vol_confirm and crsi[i] < 10.0:
                # Counter-trend only with extreme CRSI + volume
                desired_signal = SIZE * 0.5
        
        # SHORT: CRSI overbought + HTF not strongly bull (or HTF bear)
        if crsi_overbought:
            if htf_bear:
                # Strong signal: HTF bear + CRSI overbought
                desired_signal = -SIZE
            elif not htf_bull:
                # Moderate signal: HTF neutral + CRSI overbought + volume
                if vol_confirm or hma_bear:
                    desired_signal = -SIZE * 0.7
            elif htf_bull and vol_confirm and crsi[i] > 90.0:
                # Counter-trend only with extreme CRSI + volume
                desired_signal = -SIZE * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.7
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.7
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals