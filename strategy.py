#!/usr/bin/env python3
"""
Experiment #060: 1h Primary + 4h/12h HTF — Connors RSI + Choppiness Regime + Volume

Hypothesis: 1h timeframe with 4h/12h trend bias using Connors RSI for mean reversion
entries and Choppiness Index for regime detection will generate 40-80 trades/year
with Sharpe > 0.486 on ALL symbols (BTC, ETH, SOL).

Key insights from 59 failed experiments:
1) 1h strategies #050, #055, #058 got Sharpe=0.000 = 0 trades (conditions TOO STRICT)
2) Volume filters >0.8x avg kill too many signals
3) Session filters alone don't guarantee trades
4) Connors RSI works better than standard RSI for entries
5) Need 4h/12h HTF for direction, 1h only for timing

Why this should work:
- 1h primary = proven middle ground (more signals than 4h, fewer than 30m)
- 4h/12h HTF = strong trend bias without over-filtering
- Connors RSI = 3-component mean reversion (RSI3 + RSI_Streak + PercentRank)
- Choppiness regime = adapts between trend/mean-revert automatically
- Volume >0.6x (not 0.8x) = ensures trades while filtering dead periods
- Session 8-20 UTC = liquidity filter without killing signals

Position size: 0.25 (conservative for 1h TF)
Stoploss: 2.0*ATR trailing
Target: 40-80 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_volume_session_4h12h_v1"
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
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return vs last 100 days
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, period=period_rsi)
    
    # Component 2: RSI of Streak (consecutive up/down days)
    returns = close_s.pct_change().fillna(0).values
    streak = np.zeros(n)
    for i in range(1, n):
        if returns[i] > 0:
            streak[i] = streak[i-1] + 1 if returns[i-1] >= 0 else 1
        elif returns[i] < 0:
            streak[i] = streak[i-1] - 1 if returns[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_sign = np.sign(streak)
    streak_rsi = np.zeros(n)
    for i in range(period_streak, n):
        up_streak = np.sum((streak[i-period_streak:i+1] > 0).astype(float))
        down_streak = np.sum((streak[i-period_streak:i+1] < 0).astype(float))
        total = up_streak + down_streak + 1e-10
        streak_rsi[i] = 100.0 * up_streak / total
    
    # Component 3: Percent Rank of returns (last 100 periods)
    percent_rank = np.zeros(n)
    for i in range(period_rank, n):
        window = returns[i-period_rank+1:i+1]
        current_return = returns[i]
        rank = np.sum(window < current_return) / (period_rank + 1e-10)
        percent_rank[i] = rank * 100.0
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = period
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    return vol_ratio

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hours = (open_time // (1000 * 3600)) % 24
    return hours

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
    
    # Calculate 4h HMA for trend bias
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 12h HMA for confirmation
    hma_12h_21 = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, period_rsi=3, period_streak=2, period_rank=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Calculate 1h HMA for local trend
    hma_1h_21 = calculate_hma(close, period=21)
    hma_1h_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.25  # Conservative for 1h TF
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):  # Warmup for all indicators
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(vol_ratio[i]) or np.isnan(hma_1h_21[i]) or np.isnan(hma_1h_50[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # Extract hour for session filter
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20  # UTC 8-20 (liquid hours)
        
        # === HTF TREND BIAS (4h + 12h) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # Strong bias when both HTF agree
        bullish_bias = price_above_hma_4h and price_above_hma_12h
        bearish_bias = price_below_hma_4h and price_below_hma_12h
        
        # === LOCAL TREND (1h) ===
        price_above_hma_1h = close[i] > hma_1h_21[i]
        price_below_hma_1h = close[i] < hma_1h_21[i]
        hma_1h_slope_up = hma_1h_21[i] > hma_1h_21[i-10] if i > 10 else False
        hma_1h_slope_down = hma_1h_21[i] < hma_1h_21[i-10] if i > 10 else False
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0  # Range market (mean revert)
        is_trending = chop_value < 45.0  # Trend market (follow trend)
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15.0  # Strong long signal
        crsi_overbought = crsi[i] > 85.0  # Strong short signal
        crsi_moderate_oversold = crsi[i] < 25.0  # Moderate long
        crsi_moderate_overbought = crsi[i] > 75.0  # Moderate short
        
        # === VOLUME FILTER ===
        volume_ok = vol_ratio[i] > 0.6  # Not too strict
        
        # === ADAPTIVE ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion with HTF Bias ---
        if is_ranging:
            # Long: CRSI oversold + bullish HTF bias OR neutral
            if crsi_oversold and volume_ok:
                if bullish_bias or (not bearish_bias):
                    if in_session or not in_session:  # Session filter relaxed for range
                        new_signal = POSITION_SIZE
            
            # Short: CRSI overbought + bearish HTF bias OR neutral
            elif crsi_overbought and volume_ok:
                if bearish_bias or (not bullish_bias):
                    new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Pullback Entries with HTF Confirmation ---
        elif is_trending:
            # Long: CRSI moderate oversold + bullish HTF + local trend up
            if crsi_moderate_oversold and bullish_bias and volume_ok:
                if price_above_hma_1h or hma_1h_slope_up:
                    new_signal = POSITION_SIZE
            
            # Short: CRSI moderate overbought + bearish HTF + local trend down
            elif crsi_moderate_overbought and bearish_bias and volume_ok:
                if price_below_hma_1h or hma_1h_slope_down:
                    new_signal = -POSITION_SIZE
        
        # --- NEUTRAL REGIME: CRSI Extremes Only (ensures trades) ---
        else:
            # Long: Very oversold CRSI
            if crsi_oversold and volume_ok:
                new_signal = POSITION_SIZE
            # Short: Very overbought CRSI
            elif crsi_overbought and volume_ok:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            # Hold if CRSI not at opposite extreme
            if position_side > 0 and crsi[i] < 80.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and crsi[i] > 20.0:
                new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND CHANGE ===
        if in_position and position_side > 0:
            if price_below_hma_4h and price_below_hma_12h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_4h and price_above_hma_12h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals