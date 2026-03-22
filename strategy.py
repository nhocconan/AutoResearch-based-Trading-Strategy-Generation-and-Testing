#!/usr/bin/env python3
"""
Experiment #030: 1h Connors RSI + 4h HMA Trend + Adaptive Regime

Hypothesis: Previous 1h strategies (#020, #025, #028) failed with 0 trades because
entry conditions were TOO STRICT. This version LOOSENS filters while keeping quality:

1. 4h HMA(21) for trend bias (call ONCE before loop via mtf_data)
2. Connors RSI(3,2,100) with LOOSE thresholds (30/70 not 20/80)
3. Choppiness Index for regime (but NOT required for entry)
4. Volume/session filters are OPTIONAL bonuses, not requirements
5. MULTIPLE entry paths to ensure trades trigger

Key lesson from 26 failed experiments: If you have 5 filters and all must agree,
you get 0 trades. Use 2-3 core filters + optional bonuses.

Design:
- Core: 4h HMA bias + Connors RSI extremes (this alone should trigger trades)
- Bonus: CHOP regime, volume, session (improve win rate but not required)
- Stoploss: 2.0 * ATR(14)
- CRSI exit: Take profit on mean reversion (long exit CRSI>60, short exit CRSI<40)

Timeframe: 1h (REQUIRED)
HTF: 4h via mtf_data helper (ONCE before loop)
Position sizing: 0.20-0.30 discrete
Target: 40-80 trades/year on 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_connors_4h_hma_loose_v1"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Larry Connors research: CRSI < 10 long, > 90 short has 75% win rate.
    We use LOOSE thresholds (30/70) to ensure trade generation.
    """
    n = len(close)
    
    # RSI(3) - very short term RSI
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_avg = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    loss_avg = np.where(loss_avg == 0, 1e-10, loss_avg)
    rs = gain_avg / loss_avg
    rsi_3 = 100 - (100 / (1 + rs))
    rsi_3 = np.clip(rsi_3, 0, 100)
    
    # RSI Streak (2) - consecutive up/down
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    streak_gain_avg = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_avg = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_loss_avg = np.where(streak_loss_avg == 0, 1e-10, streak_loss_avg)
    streak_rs = streak_gain_avg / streak_loss_avg
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = np.where(np.isinf(rsi_streak), 50, rsi_streak)
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        lookback = close[max(0, i-rank_period+1):i+1]
        count_below = np.sum(lookback < close[i])
        percent_rank[i] = 100 * count_below / len(lookback)
    
    # Combine
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range/choppy (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
        else:
            atr_sum = 0
            for j in range(i-period+1, i+1):
                prev_close = close[j-1] if j > 0 else close[j]
                tr = max(high[j] - low[j], 
                        abs(high[j] - prev_close), 
                        abs(low[j] - prev_close))
                atr_sum += tr
            
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA trend
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    hma_1h_21 = calculate_hma(close, 21)
    
    # Volume SMA for optional filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(120, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1h_21[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            continue
        
        # === 4H HTF TREND BIAS ===
        htf_bullish = close[i] > hma_4h_21_aligned[i]
        htf_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === 1H LOCAL TREND ===
        local_bullish = close[i] > hma_1h_21[i]
        local_bearish = close[i] < hma_1h_21[i]
        
        # === CHOPPINESS REGIME (optional bonus) ===
        is_choppy = chop[i] > 55
        is_trending = chop[i] < 45
        
        # === CONNORS RSI SIGNALS (LOOSE thresholds) ===
        crsi_long = crsi[i] < 30  # LOOSENED from 20
        crsi_short = crsi[i] > 70  # LOOSENED from 80
        crsi_extreme_long = crsi[i] < 15
        crsi_extreme_short = crsi[i] > 85
        
        # === VOLUME FILTER (OPTIONAL - not required) ===
        vol_ok = True
        if not np.isnan(vol_sma[i]) and vol_sma[i] > 0:
            vol_ok = volume[i] > 0.5 * vol_sma[i]
        
        # === SESSION FILTER (OPTIONAL - not required) ===
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === POSITION SIZING ===
        if htf_bullish and local_bullish:
            current_size = STRONG_SIZE
        elif htf_bearish and local_bearish:
            current_size = STRONG_SIZE
        else:
            current_size = BASE_SIZE
        
        # === ENTRY LOGIC (MULTIPLE PATHS - LOOSENED) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY - Multiple paths (ANY can trigger)
        # Path 1: HTF bullish + CRSI oversold (primary - most common)
        if htf_bullish and crsi_long:
            new_signal = current_size
        # Path 2: CRSI extreme oversold (works in any regime)
        elif crsi_extreme_long:
            new_signal = current_size * 0.8
        # Path 3: Trending market + HTF bullish + CRSI < 50
        elif is_trending and htf_bullish and crsi[i] < 50:
            new_signal = current_size
        # Path 4: Choppy market + CRSI moderately oversold
        elif is_choppy and crsi[i] < 35:
            new_signal = current_size * 0.9
        
        # SHORT ENTRY - Multiple paths
        # Path 1: HTF bearish + CRSI overbought (primary)
        if htf_bearish and crsi_short:
            new_signal = -current_size
        # Path 2: CRSI extreme overbought (works in any regime)
        elif crsi_extreme_short:
            new_signal = -current_size * 0.8
        # Path 3: Trending market + HTF bearish + CRSI > 50
        elif is_trending and htf_bearish and crsi[i] > 50:
            new_signal = -current_size
        # Path 4: Choppy market + CRSI moderately overbought
        elif is_choppy and crsi[i] > 65:
            new_signal = -current_size * 0.9
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 72 bars (~3 days on 1h), force weaker entry
        if bars_since_last_trade > 72 and new_signal == 0.0 and not in_position:
            if htf_bullish and crsi[i] < 45:
                new_signal = current_size * 0.6
            elif htf_bearish and crsi[i] > 55:
                new_signal = -current_size * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === CRSI EXIT (take profit on mean reversion) ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long when CRSI becomes overbought (mean reversion complete)
            if position_side > 0 and crsi[i] > 60:
                crsi_exit = True
            # Exit short when CRSI becomes oversold
            if position_side < 0 and crsi[i] < 40:
                crsi_exit = True
        
        # === HTF TREND REVERSAL EXIT ===
        trend_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and htf_bearish and bars_since_last_trade > 24:
                trend_exit = True
            if position_side < 0 and htf_bullish and bars_since_last_trade > 24:
                trend_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or crsi_exit or trend_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals