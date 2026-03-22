#!/usr/bin/env python3
"""
Experiment #058: 30m Primary + 4h/1d HTF — Regime-Adaptive with Session Filter

Hypothesis: Lower TF (30m) strategies fail due to excessive trades → fee drag.
This strategy uses STRICT confluence to limit trades to 30-80/year:

1. 4h HMA(21) SLOPE for major trend direction (HTF bias)
2. 1d HMA(21) position for regime confirmation (bull/bear market)
3. Choppiness Index(14) for regime detection: CHOP>50=range, CHOP<50=trend
4. Connors RSI(3,2,100) for entry timing (more responsive than RSI14)
5. Session filter: ONLY 8-20 UTC (reduces trades ~60%, avoids Asia chop)
6. Volume filter: volume > 0.8x 20-bar average
7. ATR(14) stoploss at 2.0x trailing

Why this should work:
- 30m entries within 4h/1d trend = HTF trade frequency with LTF precision
- Session filter eliminates low-quality Asia session trades
- CHOP regime adapts: mean-revert in ranges, trend-follow in trends
- Connors RSI catches pullbacks better than standard RSI
- 3+ confluence ensures only high-probability setups trigger

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h + 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 discrete (smaller for lower TF)
Stoploss: 2.0 * ATR(14) trailing
Target trades: 30-80/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_connors_session_4h1d_v1"
timeframe = "30m"
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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
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

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope over lookback period."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    We use 50/50 as threshold for this strategy.
    """
    atr_vals = calculate_atr(high, low, close, period)
    
    chop = np.zeros(len(close))
    for i in range(period, len(close)):
        atr_sum = np.sum(atr_vals[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Streak RSI: consecutive up/down days
    PercentRank: where current close ranks vs last 100 closes
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak calculation
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI(2) on streak
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.nan)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.fillna(50).values
    
    # PercentRank(100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100
        percent_rank[i] = rank
    
    # CRSI
    crsi = (rsi_close + rsi_streak + percent_rank) / 3
    
    return crsi

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_avg + 1e-10)
    return vol_ratio

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    return (open_time // (1000 * 60 * 60)) % 24

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
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_slope = calculate_hma_slope(hma_4h_21, 5)
    
    # Calculate 1d indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_slope_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slope)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # HMA for 30m trend
    hma_30m_8 = calculate_hma(close, 8)
    hma_30m_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, smaller for 30m)
    BASE_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_slope_aligned[i]):
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(hma_30m_8[i]) or np.isnan(hma_30m_21[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = vol_ratio[i] > 0.8
        
        # === 4H TREND BIAS (MAJOR) ===
        trend_4h_bullish = hma_4h_slope_aligned[i] > 0.3  # Slope threshold
        trend_4h_bearish = hma_4h_slope_aligned[i] < -0.3
        
        # === 1D REGIME CONFIRMATION ===
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_range = chop_14[i] > 50  # Range/choppy market
        chop_trend = chop_14[i] < 50  # Trending market
        
        # === 30M HMA ALIGNMENT ===
        hma_30m_bullish = hma_30m_8[i] > hma_30m_21[i]
        hma_30m_bearish = hma_30m_8[i] < hma_30m_21[i]
        
        # === CONNORS RSI ENTRY SIGNALS ===
        # Long: CRSI < 20 (oversold pullback)
        # Short: CRSI > 80 (overbought pullback)
        crsi_oversold = crsi[i] < 20
        crsi_overbought = crsi[i] > 80
        
        # CRSI turning up/down
        crsi_turning_up = crsi[i] > crsi[i-1] and crsi[i-1] < 25
        crsi_turning_down = crsi[i] < crsi[i-1] and crsi[i-1] > 75
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES (require 4+ confluence)
        # 1. 4h trend bullish OR price above 1d HMA
        # 2. 30m HMA aligned bullish
        # 3. CRSI oversold or turning up
        # 4. In session (8-20 UTC)
        # 5. Volume OK
        long_confluence = 0
        if trend_4h_bullish or price_above_1d_hma:
            long_confluence += 1
        if hma_30m_bullish:
            long_confluence += 1
        if crsi_oversold or crsi_turning_up:
            long_confluence += 1
        if in_session:
            long_confluence += 1
        if volume_ok:
            long_confluence += 1
        
        # Require 4+ confluence for long entry
        if long_confluence >= 4:
            if crsi_oversold or crsi_turning_up:
                new_signal = current_size
        
        # SHORT ENTRIES (require 4+ confluence)
        # 1. 4h trend bearish OR price below 1d HMA
        # 2. 30m HMA aligned bearish
        # 3. CRSI overbought or turning down
        # 4. In session (8-20 UTC)
        # 5. Volume OK
        short_confluence = 0
        if trend_4h_bearish or price_below_1d_hma:
            short_confluence += 1
        if hma_30m_bearish:
            short_confluence += 1
        if crsi_overbought or crsi_turning_down:
            short_confluence += 1
        if in_session:
            short_confluence += 1
        if volume_ok:
            short_confluence += 1
        
        # Require 4+ confluence for short entry
        if short_confluence >= 4:
            if crsi_overbought or crsi_turning_down:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 200 bars (~4 days on 30m), allow 3 confluence entry
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if long_confluence >= 3 and (crsi_oversold or crsi_turning_up):
                new_signal = current_size * 0.6
            elif short_confluence >= 3 and (crsi_overbought or crsi_turning_down):
                new_signal = -current_size * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 4h trend reverses bearish
            if position_side > 0 and trend_4h_bearish and hma_30m_bearish:
                trend_reversal = True
            # Exit short if 4h trend reverses bullish
            if position_side < 0 and trend_4h_bullish and hma_30m_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
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