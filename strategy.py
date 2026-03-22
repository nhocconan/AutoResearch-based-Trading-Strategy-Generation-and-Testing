#!/usr/bin/env python3
"""
Experiment #010: 1h Connors RSI + 4h/12h HMA Trend + Choppiness Regime Filter

Hypothesis: Lower timeframe (1h) with strict HTF filters will capture entries
within major trends while avoiding whipsaw. Key design:
1. 12h HMA(21) for major trend direction (call ONCE before loop via mtf_data)
2. 4h HMA(21) for intermediate trend confirmation (call ONCE before loop)
3. Choppiness Index(14) for regime detection: CHOP>55=range, CHOP<45=trend
4. Connors RSI for precise mean-reversion entries within trend
5. Session filter (8-20 UTC) for better liquidity
6. Volume filter (>0.8x 20-bar avg) to confirm moves
7. Asymmetric sizing: 0.30 strong trend, 0.20 weak trend
8. ATR(14) stoploss at 2.5x

Why this should work:
- Connors RSI has 75% win rate on mean reversion (proven in literature)
- CHOP regime filter prevents trend strategies in ranges (major failure mode)
- Dual HTF (4h+12h) ensures we only trade with major trend
- Session/volume filters reduce false signals during low liquidity
- 1h TF targets 30-60 trades/year (optimal for fee efficiency)
- Discrete sizing minimizes fee churn on signal changes

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h and 12h via mtf_data helper (call ONCE before loop each)
Position sizing: 0.20-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_connors_4h_12h_hma_chop_session_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_avg = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    loss_avg = np.where(loss_avg == 0, 1e-10, loss_avg)
    rs = gain_avg / loss_avg
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar (simplified: high-low)
    tr = high - low
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_val = highest_high - lowest_low
        
        if range_val > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / range_val) / np.log10(period)
        else:
            chop[i] = 50  # neutral
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(close, 3): 3-period RSI on close
    RSI(streak, 2): 2-period RSI on up/down streak length
    PercentRank(close, 100): percentile rank of close over last 100 bars
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI(2) on streak (convert streak to gains/losses)
    streak_delta = np.diff(streak)
    streak_delta = np.insert(streak_delta, 0, 0)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0)
    
    streak_gain_avg = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_avg = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_avg = np.where(streak_loss_avg == 0, 1e-10, streak_loss_avg)
    streak_rsi = 100 - (100 / (1 + streak_gain_avg / streak_loss_avg))
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # PercentRank(100)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        crsi[i] = (rsi_close[i] + streak_rsi[i] + rank) / 3
    
    # Fill early bars
    crsi[:rank_period] = crsi[rank_period] if n > rank_period else 50
    
    return np.clip(crsi, 0, 100)

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    return (open_time // (1000 * 3600)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HTF HMA trends
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF indicators to 1h timeframe (auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    vol_avg_20 = calculate_volume_avg(volume, 20)
    
    # Also calculate 1h HMA for local trend
    hma_1h_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    WEAK_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        # === 12H HTF TREND BIAS (MAJOR TREND) ===
        htf_12h_bullish = close[i] > hma_12h_aligned[i]
        htf_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # === 4H HTF TREND BIAS (INTERMEDIATE TREND) ===
        htf_4h_bullish = close[i] > hma_4h_aligned[i]
        htf_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === 1H LOCAL TREND ===
        local_bullish = close[i] > hma_1h_21[i]
        local_bearish = close[i] < hma_1h_21[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range (use mean reversion)
        # CHOP < 45 = trend (use trend following)
        # 45-55 = neutral (reduce size or skip)
        regime_range = chop_14[i] > 55
        regime_trend = chop_14[i] < 45
        regime_neutral = not regime_range and not regime_trend
        
        # === CONNORS RSI SIGNALS ===
        # Long: CRSI < 15 (oversold)
        # Short: CRSI > 85 (overbought)
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        
        # === VOLUME FILTER ===
        # Volume must be > 0.8x average to confirm move
        volume_ok = volume[i] > 0.8 * vol_avg_20[i] if not np.isnan(vol_avg_20[i]) else True
        
        # === SESSION FILTER (8-20 UTC) ===
        # Only trade during high liquidity hours
        hour = get_hour_from_open_time(open_time[i])
        session_ok = 8 <= hour <= 20
        
        # === POSITION SIZING BASED ON TREND CONFLUENCE ===
        # Strong: 12h + 4h + 1h all agree
        # Medium: 12h + 4h agree
        # Weak: only 12h
        if htf_12h_bullish and htf_4h_bullish and local_bullish:
            current_size = STRONG_SIZE
        elif htf_12h_bullish and htf_4h_bullish:
            current_size = BASE_SIZE
        elif htf_12h_bullish:
            current_size = WEAK_SIZE
        elif htf_12h_bearish and htf_4h_bearish and local_bearish:
            current_size = STRONG_SIZE
        elif htf_12h_bearish and htf_4h_bearish:
            current_size = BASE_SIZE
        elif htf_12h_bearish:
            current_size = WEAK_SIZE
        else:
            current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: 12h bullish + CRSI oversold + session + volume
        # In range regime: mean reversion long
        # In trend regime: pullback long within uptrend
        if htf_12h_bullish and crsi_oversold and session_ok and volume_ok:
            if regime_range or regime_trend:
                new_signal = current_size
        
        # SHORT ENTRY: 12h bearish + CRSI overbought + session + volume
        if htf_12h_bearish and crsi_overbought and session_ok and volume_ok:
            if regime_range or regime_trend:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 50 bars (~2 days on 1h), allow weaker entry
        if bars_since_last_trade > 50 and new_signal == 0.0 and not in_position:
            if htf_12h_bullish and crsi[i] < 25 and session_ok:
                new_signal = current_size * 0.8
            elif htf_12h_bearish and crsi[i] > 75 and session_ok:
                new_signal = -current_size * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 12h trend turns bearish
            if position_side > 0 and htf_12h_bearish:
                trend_reversal = True
            # Exit short if 12h trend turns bullish
            if position_side < 0 and htf_12h_bullish:
                trend_reversal = True
        
        # === CRSI EXTREME EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long when CRSI becomes very overbought (>80)
            if position_side > 0 and crsi[i] > 80:
                crsi_exit = True
            # Exit short when CRSI becomes very oversold (<20)
            if position_side < 0 and crsi[i] < 20:
                crsi_exit = True
        
        # Apply stoploss or reversals
        if stoploss_triggered or trend_reversal or crsi_exit:
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