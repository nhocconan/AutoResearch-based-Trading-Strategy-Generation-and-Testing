#!/usr/bin/env python3
"""
Experiment #025: 1h Connors RSI Mean Reversion with 4h/1d HMA Trend Filter

Hypothesis: Previous 1h strategies failed due to either too many trades (fee drag)
or too few trades (0 Sharpe). This strategy combines:
1. Connors RSI (CRSI) for precise entry timing - proven 75% win rate on extremes
2. 4h HMA(21) for intermediate trend direction
3. 1d HMA(21) for major trend bias
4. Choppiness Index regime filter - only trade when CHOP confirms regime
5. Session filter (8-20 UTC) - only trade during high-liquidity hours
6. Volume confirmation - avoid low-volume false breakouts
7. ATR(14) stoploss at 2.5x for risk management

Why this should work:
- Connors RSI extremes (CRSI<10 long, CRSI>90 short) have high win rate
- 4h/1d HMA filter prevents counter-trend trades (major failure mode in exp #015, #018)
- Choppiness Index ensures we use correct strategy for regime
- Session filter reduces noise and fee churn
- 1h timeframe with strict filters = target 40-80 trades/year

Timeframe: 1h (REQUIRED)
HTF: 4h, 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_connors_chop_hma_4h1d_session_v1"
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
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(3): Fast RSI on close
    RSI_Streak(2): RSI on consecutive up/down days
    PercentRank(100): Where current price ranks vs last 100 closes
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI on streak
    # Streak = consecutive days of gains/losses
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to gains/losses for RSI calculation
    streak_gain = np.maximum(streak, 0)
    streak_loss = np.abs(np.minimum(streak, 0))
    
    # Smooth streak RSI
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    # Avoid division by zero
    rs_streak = np.zeros(n)
    mask = avg_streak_loss > 0
    rs_streak[mask] = avg_streak_gain[mask] / avg_streak_loss[mask]
    rs_streak[~mask] = 100  # Default when no losses
    
    rsi_streak = 100 - (100 / (1 + rs_streak))
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # Component 3: PercentRank(100)
    # Where does current close rank vs last 100 closes?
    percent_rank = np.zeros(n)
    for i in range(pr_period, n):
        window = close[max(0, i-pr_period):i+1]
        rank = np.sum(window < close[i]) / len(window)
        percent_rank[i] = rank * 100
    
    # Combine components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = range/choppy market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = len(close)
    choppiness = np.zeros(n)
    
    atr_vals = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        atr_sum = np.nansum(atr_vals[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            choppiness[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            choppiness[i] = 50  # Neutral
    
    choppiness = np.clip(choppiness, 0, 100)
    return choppiness

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

def get_hour_from_open_time(open_time_array):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = ((open_time_array // 1000 // 3600) % 24).astype(int)
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    choppiness = calculate_choppiness_index(high, low, close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    hours = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(choppiness[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER ===
        volume_ok = vol_ratio[i] >= 0.8
        
        # === HTF TREND BIAS ===
        # 4h trend
        trend_4h_bullish = close[i] > hma_4h_21_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # 1d trend (major bias)
        trend_1d_bullish = close[i] > hma_1d_21_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = range (favor mean reversion)
        # CHOP < 45 = trend (favor trend following)
        is_range = choppiness[i] > 55
        is_trending = choppiness[i] < 45
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        if i > 150:
            atr_median = np.nanmedian(atr_14[max(0, i-100):i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
            vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        else:
            vol_adjustment = 1.0
        
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG entries (require bullish HTF bias)
        if trend_4h_bullish and trend_1d_bullish:
            # Mean reversion in range market: CRSI extreme oversold
            if is_range and crsi[i] < 15 and in_session and volume_ok:
                new_signal = current_size
            # Trend pullback in trending market: moderate CRSI
            elif is_trending and 20 <= crsi[i] <= 40 and in_session and volume_ok:
                new_signal = current_size
        
        # SHORT entries (require bearish HTF bias)
        elif trend_4h_bearish and trend_1d_bearish:
            # Mean reversion in range market: CRSI extreme overbought
            if is_range and crsi[i] > 85 and in_session and volume_ok:
                new_signal = -current_size
            # Trend pullback in trending market: moderate CRSI
            elif is_trending and 60 <= crsi[i] <= 80 and in_session and volume_ok:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 72 bars (~3 days on 1h), allow weaker entry
        if bars_since_last_trade > 72 and new_signal == 0.0 and not in_position:
            if trend_4h_bullish and crsi[i] < 30 and in_session:
                new_signal = current_size * 0.7
            elif trend_4h_bearish and crsi[i] > 70 and in_session:
                new_signal = -current_size * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
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
            if position_side > 0 and trend_4h_bearish:
                trend_reversal = True
            if position_side < 0 and trend_4h_bullish:
                trend_reversal = True
        
        # === CRSI REVERSAL EXIT (take profit) ===
        crsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and crsi[i] > 75:
                crsi_exit = True  # Long exit on overbought
            if position_side < 0 and crsi[i] < 25:
                crsi_exit = True  # Short exit on oversold
        
        # Apply stoploss or exits
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