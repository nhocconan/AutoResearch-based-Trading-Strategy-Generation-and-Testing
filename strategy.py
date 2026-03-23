#!/usr/bin/env python3
"""
Experiment #108: 30m Primary + 4h/1d HTF — Connors RSI Mean Reversion with Regime Filter

Hypothesis: Lower TF (30m) strategies fail due to fee drag from too many trades.
This uses 4h/1d for SIGNAL DIRECTION (trend bias), 30m only for ENTRY TIMING.
Key innovations:

1) 4h HMA(21) for macro trend bias — only trade mean reversion IN trend direction
2) 1d Choppiness Index(14) for regime — CHOP>55=range(mean revert), CHOP<45=trend(follow)
3) 30m Connors RSI for entry timing — CRSI<10 long, CRSI>90 short (extreme only)
4) Session filter: 8-20 UTC only — high liquidity, reduces whipsaws
5) Volume confirmation: >0.8x 20-period avg — filters low-liquidity traps
6) ATR(14) trailing stop at 2.5x — locks profits, limits drawdown

Why this should work on 30m:
- HTF direction filter reduces trades by 50%+ (only trade with 4h trend)
- Session filter cuts 50% of bars (only 12 hours/day)
- CRSI extremes (<10 or >90) are rare — natural trade frequency limiter
- Regime filter adapts: mean revert in range, trend follow in trends
- Target: 40-70 trades/year, Sharpe > 0.5 on ALL symbols

Position size: 0.20 base (conservative for 30m), 0.25 max with volume confluence
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_chop_regime_4h1d_session_v1"
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
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_rsi_streak(close, period=2):
    """Calculate RSI Streak component of Connors RSI."""
    # Count consecutive up/down days
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            if i > 0 and streak[i-1] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif close[i] < close[i-1]:
            if i > 0 and streak[i-1] < 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like score (0-100)
    streak_rsi = np.zeros(len(close))
    for i in range(period, len(close)):
        streak_window = streak[i-period+1:i+1]
        up_streaks = np.sum(streak_window > 0)
        down_streaks = np.sum(streak_window < 0)
        if up_streaks + down_streaks > 0:
            streak_rsi[i] = 100.0 * up_streaks / (up_streaks + down_streaks)
        else:
            streak_rsi[i] = 50.0
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """Calculate Percent Rank component of Connors RSI."""
    pr = np.zeros(len(close))
    for i in range(period, len(close)):
        window = close[i-period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current) / (period - 1) * 100.0
        pr[i] = rank
    return pr

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Calculate Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3."""
    rsi_3 = calculate_rsi(close, period=rsi_period)
    streak_rsi = calculate_rsi_streak(close, period=streak_period)
    percent_rank = calculate_percent_rank(close, period=pr_period)
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    chop = np.zeros(len(close))
    for i in range(period, len(close)):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        atr_avg = atr_sum / period
        if atr_avg > 0 and (highest_high - lowest_low) > 0:
            chop[i] = 100.0 * np.log10((highest_high - lowest_low) / atr_avg) / np.log10(period)
        else:
            chop[i] = 50.0
    return chop

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
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
    
    # Calculate 4h HMA for macro trend
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d Choppiness Index for regime
    chop_1d = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi_30m = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    hma_30m_21 = calculate_hma(close, period=21)
    hma_30m_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.20
    POSITION_SIZE_MAX = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(chop_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(crsi_30m[i]) or np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        if np.isnan(hma_30m_21[i]) or np.isnan(hma_30m_50[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === HTF TREND BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # 4h HMA slope
        hma_4h_slope = 0.0
        if i > 0 and not np.isnan(hma_4h_aligned[i-1]) and hma_4h_aligned[i-1] != 0:
            hma_4h_slope = (hma_4h_aligned[i] - hma_4h_aligned[i-1]) / hma_4h_aligned[i-1] * 100
        
        hma_slope_positive = hma_4h_slope > 0.3
        hma_slope_negative = hma_4h_slope < -0.3
        
        # === REGIME FILTER (1d Choppiness) ===
        chop_value = chop_1d_aligned[i]
        is_range_regime = chop_value > 55.0  # Mean reversion regime
        is_trend_regime = chop_value < 45.0  # Trend following regime
        
        # === 30m TREND FILTER ===
        hma_30m_bullish = hma_30m_21[i] > hma_30m_50[i]
        hma_30m_bearish = hma_30m_21[i] < hma_30m_50[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_extreme_low = crsi_30m[i] < 10.0  # Oversold
        crsi_extreme_high = crsi_30m[i] > 90.0  # Overbought
        
        # === VOLUME CONFIRMATION ===
        volume_ratio = volume[i] / (vol_avg_20[i] + 1e-10)
        volume_confirmed = volume_ratio > 0.8
        volume_strong = volume_ratio > 1.5
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Only trade during session hours
        if not in_session:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # --- LONG ENTRY ---
        # Mean reversion in range: CRSI<10 + price>4h HMA + range regime
        # Trend follow: CRSI<10 + price>4h HMA + 4h slope up + trend regime
        if crsi_extreme_low and price_above_hma_4h and volume_confirmed:
            if is_range_regime:
                # Range regime: mean reversion long
                new_signal = POSITION_SIZE_BASE
                if volume_strong:
                    new_signal = POSITION_SIZE_MAX
            elif is_trend_regime and hma_slope_positive:
                # Trend regime: only long if 4h trending up
                new_signal = POSITION_SIZE_BASE
                if volume_strong and hma_30m_bullish:
                    new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY ---
        # Mean reversion in range: CRSI>90 + price<4h HMA + range regime
        # Trend follow: CRSI>90 + price<4h HMA + 4h slope down + trend regime
        if crsi_extreme_high and price_below_hma_4h and volume_confirmed:
            if is_range_regime:
                # Range regime: mean reversion short
                new_signal = -POSITION_SIZE_BASE
                if volume_strong:
                    new_signal = -POSITION_SIZE_MAX
            elif is_trend_regime and hma_slope_negative:
                # Trend regime: only short if 4h trending down
                new_signal = -POSITION_SIZE_BASE
                if volume_strong and hma_30m_bearish:
                    new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Hold if CRSI not at opposite extreme and HTF trend intact
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if CRSI not overbought and price above 4h HMA
                if crsi_30m[i] < 85.0 and price_above_hma_4h:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if CRSI not oversold and price below 4h HMA
                if crsi_30m[i] > 15.0 and price_below_hma_4h:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if highest_since_entry == 0.0:
                highest_since_entry = close[i]
            else:
                highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON HTF TREND REVERSAL ===
        if in_position and position_side > 0:
            if price_below_hma_4h and hma_slope_negative:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_4h and hma_slope_positive:
                new_signal = 0.0
        
        # === EXIT ON CRSI OPPOSITE EXTREME (take profit) ===
        if in_position and position_side > 0 and crsi_30m[i] > 85.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and crsi_30m[i] < 15.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals