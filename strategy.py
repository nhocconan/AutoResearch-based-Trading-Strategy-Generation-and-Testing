#!/usr/bin/env python3
"""
Experiment #760: 1h Primary + 4h/12h HTF — CRSI + Choppiness + Dual HMA + Session/Volume

Hypothesis: After analyzing 500+ failed strategies and the success of #751 (Sharpe=0.342):
1. 1h timeframe needs VERY strict filters to avoid fee drag (>100 trades/yr = failure)
2. 12h HMA(21) provides macro trend bias (proven in multi-TF strategies)
3. 4h Choppiness Index filters regime: CHOP>55=range (mean revert), CHOP<45=trend (follow)
4. 1h CRSI(3,2,100) for precise entry timing at extremes (<15 long, >85 short)
5. Session filter (8-20 UTC) reduces trades to 40-80/yr target
6. Volume filter (>0.8x 20-bar avg) confirms participation
7. Looser CRSI thresholds (<20/>80) ensure >=30 trades/train (fixes #750/#758 zero-trade issue)

Strategy design:
1. 12h HMA(21) for macro trend bias (aligned via mtf_data helper)
2. 4h HMA(21) + Choppiness(14) for intermediate trend + regime
3. 1h CRSI for entry timing (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
4. Session filter: only trade 8-20 UTC (high liquidity hours)
5. Volume filter: volume > 0.8 * SMA20(volume)
6. ATR(14) trailing stop 2.5x protects against adverse moves
7. Discrete signals: 0.0, ±0.25, ±0.30

Key improvements from failed 1h strategies (#750, #755, #758):
- Added 12h HTF for stronger trend bias (not just 4h)
- Looser CRSI thresholds (<20/>80 instead of <15/>85) to ensure trades
- Session + volume filters reduce trade count to target 40-80/yr
- Better hold logic to maintain positions through trends
- Fixed zero-trade issue by ensuring entry conditions can actually trigger

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 40-80 trades/year with strict filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_dual_hma_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(series, period):
    """Hull Moving Average - smoother and more responsive than EMA."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_rsi_streak(close, period=2):
    """
    RSI Streak Component of Connors RSI.
    Measures consecutive up/down bars.
    """
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    if n < period + 5:
        return streak_rsi
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    abs_streak = np.abs(streak)
    max_streak = np.max(abs_streak[~np.isnan(abs_streak)]) if np.any(~np.isnan(abs_streak)) else 1
    
    if max_streak > 0:
        streak_score = 100 * abs_streak / max_streak
    else:
        streak_score = np.zeros(n)
    
    streak_rsi = np.where(streak >= 0, streak_score, 100 - streak_score)
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank Component of Connors RSI.
    Measures where current close ranks vs previous N closes.
    """
    n = len(close)
    pct_rank = np.full(n, np.nan)
    
    if n < period:
        return pct_rank
    
    for i in range(period, n):
        window = close[i-period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current)
        pct_rank[i] = 100 * rank / (period - 1)
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion.
    Long: CRSI < 15-20
    Short: CRSI > 80-85
    """
    rsi_fast = calculate_rsi(close, period=rsi_period)
    streak_rsi = calculate_rsi_streak(close, period=streak_period)
    pct_rank = calculate_percent_rank(close, period=rank_period)
    
    crsi = (rsi_fast + streak_rsi + pct_rank) / 3
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures whether market is trending or ranging.
    CHOP > 55 = ranging (mean reversion)
    CHOP < 45 = trending (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_sma(close, period):
    """Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    return (open_time // 3600000) % 24

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
    
    # Calculate primary (1h) indicators
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_1h = calculate_atr(high, low, close, period=14)
    chop_1h = calculate_choppiness(high, low, close, period=14)
    vol_sma20_1h = calculate_sma(volume, period=20)
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 4h Choppiness for regime detection
    chop_4h_raw = calculate_choppiness(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, period=14)
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(chop_4h_aligned[i]) or np.isnan(vol_sma20_1h[i]):
            continue
        if vol_sma20_1h[i] <= 1e-10:
            continue
        
        # Extract UTC hour for session filter
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20  # High liquidity hours
        
        # Volume filter
        volume_ok = volume[i] > 0.8 * vol_sma20_1h[i]
        
        # === TREND BIAS (12h + 4h HTF HMA) ===
        trend_12h_bullish = close[i] > hma_12h_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_aligned[i]
        
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # Strong trend alignment (both 12h and 4h agree)
        strong_bullish = trend_12h_bullish and trend_4h_bullish
        strong_bearish = trend_12h_bearish and trend_4h_bearish
        
        # === REGIME DETECTION (4h Choppiness Index) ===
        trending_regime = chop_4h_aligned[i] < 45
        ranging_regime = chop_4h_aligned[i] > 55
        
        # === CRSI SIGNALS (Connors RSI) - LOOSENED for trade frequency ===
        crsi_extreme_low = crsi_1h[i] < 20  # Was <15, loosened for more trades
        crsi_extreme_high = crsi_1h[i] > 80  # Was >85, loosened for more trades
        crsi_oversold = crsi_1h[i] < 30
        crsi_overbought = crsi_1h[i] > 70
        
        desired_signal = 0.0
        
        # Only trade during high-liquidity session with volume confirmation
        if in_session and volume_ok:
            # === TRENDING REGIME LOGIC (CHOP < 45) ===
            if trending_regime:
                # Long: Strong bullish trend + CRSI pullback
                if strong_bullish and crsi_oversold:
                    desired_signal = BASE_SIZE
                
                # Short: Strong bearish trend + CRSI rally
                if strong_bearish and crsi_overbought:
                    desired_signal = -BASE_SIZE
                
                # Trend continuation (less strict)
                if trend_4h_bullish and crsi_1h[i] > 35 and crsi_1h[i] < 65:
                    desired_signal = REDUCED_SIZE
                
                if trend_4h_bearish and crsi_1h[i] > 35 and crsi_1h[i] < 65:
                    desired_signal = -REDUCED_SIZE
            
            # === RANGING REGIME LOGIC (CHOP > 55) ===
            elif ranging_regime:
                # Mean reversion long: CRSI extreme low + not strongly bearish
                if crsi_extreme_low and not strong_bearish:
                    desired_signal = REDUCED_SIZE
                
                # Mean reversion short: CRSI extreme high + not strongly bullish
                if crsi_extreme_high and not strong_bullish:
                    desired_signal = -REDUCED_SIZE
            
            # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
            else:
                # Conservative: only enter on CRSI extremes + trend alignment
                if crsi_extreme_low and strong_bullish:
                    desired_signal = REDUCED_SIZE
                
                if crsi_extreme_high and strong_bearish:
                    desired_signal = -REDUCED_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend still bullish and CRSI not overbought
                if trend_4h_bullish and crsi_1h[i] < 75:
                    desired_signal = BASE_SIZE if trending_regime else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if trend still bearish and CRSI not oversold
                if trend_4h_bearish and crsi_1h[i] > 25:
                    desired_signal = -BASE_SIZE if trending_regime else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses or CRSI very overbought
            if strong_bearish and crsi_1h[i] > 70:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses or CRSI very oversold
            if strong_bullish and crsi_1h[i] < 30:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
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
        
        signals[i] = desired_signal
    
    return signals