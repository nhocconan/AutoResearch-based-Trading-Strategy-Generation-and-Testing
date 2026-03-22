#!/usr/bin/env python3
"""
Experiment #325: 1h Primary + 4h/1d HTF — Regime-Adaptive Connors RSI + HMA Trend + Session Filter

Hypothesis: A regime-adaptive strategy using Connors RSI for entries, 4h HMA for trend direction,
and 1d Choppiness Index for regime detection will outperform pure trend or pure mean-reversion.

Why this might work:
1. Connors RSI (CRSI) has 75% win rate for mean reversion - catches pullbacks in trends
2. 1d Choppiness Index tells us WHEN to trend-follow vs mean-revert (regime filter)
3. 4h HMA(21) provides major trend direction without excessive lag
4. Session filter (8-20 UTC) avoids Asian session noise and low-volume whipsaws
5. Volume confirmation (>0.8x 20-bar avg) ensures moves have participation
6. Asymmetric sizing: longs 0.25, shorts 0.20 (crypto bias)

Key innovation: Different logic per regime
- CHOP < 45 (trending): Follow 4h HMA direction, enter on CRSI pullbacks (30-50 long, 50-70 short)
- CHOP > 55 (ranging): Mean revert at extremes (CRSI < 15 long, CRSI > 85 short)
- CHOP 45-55 (transition): Reduce position size by 30%

Target: 40-70 trades/year on 1h (appropriate for hourly with strict filters)
Position sizing: 0.20-0.25 base, max 0.30 strong conviction
Stoploss: 2.5 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_hma_chop_4h1d_session_v1"
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
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    More responsive than EMA with less lag.
    """
    n = period
    n_half = n // 2
    n_sqrt = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    # WMA helper
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, n_half)
    wma_full = wma(close_s, n)
    
    diff = 2.0 * wma_half - wma_full
    hma = wma(diff, n_sqrt)
    
    return hma.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(close, 3): 3-period RSI on price
    RSI(streak, 2): 2-period RSI on up/down streak length
    PercentRank(close, 100): Where current close ranks vs last 100 closes (0-100)
    
    CRSI < 10 = extremely oversold (long opportunity)
    CRSI > 90 = extremely overbought (short opportunity)
    CRSI 30-50 = pullback long in uptrend
    CRSI 50-70 = pullback short in downtrend
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3) on price
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_price = 100.0 - (100.0 / (1.0 + rs))
    
    # Component 2: RSI(2) on streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: PercentRank(100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        count_below = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_below / rank_period
    
    # Combine into CRSI
    crsi = (rsi_price.values + rsi_streak.values + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
    """
    n = period
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Calculate ATR
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    
    atr_sum = atr.rolling(window=n, min_periods=n).sum()
    hh = high_s.rolling(window=n, min_periods=n).max()
    ll = low_s.rolling(window=n, min_periods=n).min()
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh.iloc[i] - ll.iloc[i]
        if range_hl > 0 and atr_sum.iloc[i] > 0:
            chop[i] = 100 * np.log10(atr_sum.iloc[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def extract_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hours = (open_time // 3600000) % 24
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
    
    # Calculate 4h HTF indicators (major trend direction)
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 1d HTF indicators (regime detection)
    chop_1d_14 = calculate_choppiness_index(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        14
    )
    chop_1d_14_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_14)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_1h_21 = calculate_hma(close, 21)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    vol_avg_20 = calculate_volume_avg(volume, 20)
    
    # Extract UTC hours for session filter
    hours = np.array([extract_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Smaller for 1h timeframe to reduce fee impact
    LONG_BASE = 0.22
    LONG_STRONG = 0.28
    SHORT_BASE = 0.18
    SHORT_STRONG = 0.24
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]):
            continue
        
        if np.isnan(chop_1d_14_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(hma_1h_21[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        # Avoid Asian session (0-8 UTC) and late night (20-24 UTC)
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER ===
        vol_ratio = volume[i] / (vol_avg_20[i] + 1e-10)
        volume_ok = vol_ratio > 0.7  # At least 70% of average volume
        
        # === 1D REGIME (Choppiness Index) ===
        chop = chop_1d_14_aligned[i]
        regime_trend = chop < 45.0  # Trending market
        regime_range = chop > 55.0  # Ranging market
        regime_transition = 45.0 <= chop <= 55.0  # Transition
        
        # === 4H TREND DIRECTION ===
        trend_bull = close[i] > hma_4h_21_aligned[i]
        trend_bear = close[i] < hma_4h_21_aligned[i]
        
        # === 1H LOCAL TREND ===
        local_bull = close[i] > hma_1h_21[i]
        local_bear = close[i] < hma_1h_21[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        crsi_pullback_long = 28.0 < crsi[i] < 48.0
        crsi_pullback_short = 52.0 < crsi[i] < 72.0
        crsi_rising = crsi[i] > crsi[i-1] if i > 0 else False
        crsi_falling = crsi[i] < crsi[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # Only trade during session and with volume
        if in_session and volume_ok:
            # TRENDING REGIME (CHOP < 45): Follow 4h trend, enter on pullbacks
            if regime_trend:
                # Long: 4h bull + CRSI pullback + local bull confirmation
                if trend_bull and crsi_pullback_long and local_bull and crsi_rising:
                    new_signal = LONG_BASE
                
                # Long: 4h bull + CRSI extreme oversold (strong conviction)
                elif trend_bull and crsi_extreme_oversold:
                    new_signal = LONG_STRONG
                
                # Short: 4h bear + CRSI pullback + local bear confirmation
                elif trend_bear and crsi_pullback_short and local_bear and crsi_falling:
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE
                
                # Short: 4h bear + CRSI extreme overbought (strong conviction)
                elif trend_bear and crsi_extreme_overbought:
                    if new_signal == 0.0:
                        new_signal = -SHORT_STRONG
            
            # RANGING REGIME (CHOP > 55): Mean revert at extremes
            elif regime_range:
                # Long: CRSI extreme oversold in range
                if crsi_extreme_oversold:
                    new_signal = LONG_BASE * 0.9
                
                # Short: CRSI extreme overbought in range
                elif crsi_extreme_overbought:
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE * 0.9
            
            # TRANSITION REGIME (CHOP 45-55): Reduced size, wait for clarity
            elif regime_transition:
                # Only take extreme CRSI signals with reduced size
                if crsi_extreme_oversold and trend_bull:
                    new_signal = LONG_BASE * 0.6
                elif crsi_extreme_overbought and trend_bear:
                    if new_signal == 0.0:
                        new_signal = -SHORT_BASE * 0.6
        
        # === FREQUENCY SAFEGUARD (ensure 40+ trades/year on 1h) ===
        # Force trade if no signal for 120 bars (~5 days) and conditions are reasonable
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_bull and crsi[i] < 40.0 and in_session:
                new_signal = LONG_BASE * 0.5
            elif trend_bear and crsi[i] > 60.0 and in_session:
                new_signal = -SHORT_BASE * 0.5
        
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
        
        # === CRSI REVERSAL EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Long position: exit when CRSI turns overbought
            if position_side > 0 and crsi_extreme_overbought:
                crsi_exit = True
            # Short position: exit when CRSI turns oversold
            if position_side < 0 and crsi_extreme_oversold:
                crsi_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_exit = False
        if in_position and position_side != 0:
            # Long position: exit when 4h trend turns bearish
            if position_side > 0 and trend_bear and local_bear:
                trend_exit = True
            # Short position: exit when 4h trend turns bullish
            if position_side < 0 and trend_bull and local_bull:
                trend_exit = True
        
        # === SESSION EXIT (close position before session end) ===
        session_exit = False
        if in_position and position_side != 0:
            # Exit if we're at hour 19 and still in position (avoid overnight)
            if hours[i] >= 19:
                session_exit = True
        
        if stoploss_triggered or crsi_exit or trend_exit:
            new_signal = 0.0
        
        # === DISCRETIZE SIGNAL (reduce churn) ===
        if new_signal != 0.0:
            if abs(new_signal) < 0.10:
                new_signal = 0.0
            elif new_signal > 0.26:
                new_signal = LONG_STRONG
            elif new_signal > 0:
                new_signal = LONG_BASE
            elif new_signal < -0.22:
                new_signal = -SHORT_STRONG
            else:
                new_signal = -SHORT_BASE
        
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