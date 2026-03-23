#!/usr/bin/env python3
"""
Experiment #795: 1h Primary + 4h/1d HTF — Connors RSI Mean Reversion + Choppiness Regime + Session Filter

Hypothesis: After 540+ failed strategies, lower TF (1h) requires EXTREME selectivity:
1. Connors RSI (CRSI) has 75% win rate for mean reversion vs regular RSI's 55%
2. CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — catches short-term extremes
3. 4h HMA(21) for trend bias — smoother than 1h, avoids whipsaw
4. 1d Choppiness Index for regime — CHOP>55=range (mean revert), CHOP<45=trend (follow)
5. Session filter (8-20 UTC) — only trade during high liquidity, reduces trades 60%
6. Volume filter (>0.8x avg) — confirms moves, avoids fakeouts
7. Asymmetric sizing: 0.25 base, 0.30 high confidence (regime + volume + session)
8. ATR(14) trailing stop at 2.5x — protects from major drawdowns
9. Target: 30-60 trades/year on 1h (strict filters to avoid fee drag)

Key differences from failed 1h strategies (#785, #790):
- CRSI instead of RSI(14) — more sensitive to short-term extremes
- Session filter MANDATORY for 1h (reduces trades from 200→50/year)
- 4h HMA trend bias (not 1h EMA) — HTF direction, LTF timing
- Relaxed CRSI thresholds: <20/>80 (not <10/>90) — ensures trades
- Hold logic: maintain position until opposite signal or stoploss
- Volume filter: 0.8x (not 1.5x) — less restrictive for 1h

Strategy design:
1. 4h HMA(21) aligned via mtf_data for trend bias
2. 1d Choppiness(14) aligned for regime detection
3. 1h CRSI(3,2,100) for entry timing
4. 1h ATR(14) for trailing stop (2.5x)
5. Session filter: hour 8-20 UTC only
6. Volume filter: >0.8x 20-period SMA
7. Discrete signals: 0.0, ±0.25, ±0.30
8. Stoploss: signal→0 when price moves 2.5*ATR against position

Target: Sharpe > 0.612, trades >= 10 train, >= 3 test, ALL symbols positive
Timeframe: 1h (target 30-60 trades/year with session filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_hma_4h1d_session_atr_v1"
timeframe = "1h"
leverage = 1.0

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
    RSI of Streak Length — measures consecutive up/down bars.
    Streak: count consecutive gains or losses.
    Then calculate RSI on streak values.
    """
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    if n < period + 5:
        return streak_rsi
    
    # Calculate streak length
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert to absolute streak for RSI calculation
    abs_streak = np.abs(streak)
    
    # Calculate RSI on streak (using gains when streak positive, losses when negative)
    delta = np.diff(abs_streak)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        streak_rsi = 100 - (100 / (1 + rs))
    
    streak_rsi = np.clip(streak_rsi, 0, 100)
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percentile Rank of returns over last N periods.
    Returns value 0-100 indicating where current return ranks.
    """
    n = len(close)
    pr = np.full(n, np.nan)
    
    if n < period + 1:
        return pr
    
    returns = np.diff(close) / (close[:-1] + 1e-10)
    
    for i in range(period, n):
        window = returns[i-period:i]
        current_return = returns[i-1] if i > 0 else 0
        
        if len(window) > 0:
            rank = np.sum(window < current_return) / len(window)
            pr[i] = rank * 100
        else:
            pr[i] = 50
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries.
    """
    rsi = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    with np.errstate(invalid='ignore'):
        crsi = (rsi + streak_rsi + pr) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

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
    Choppiness Index — measures market choppy vs trending.
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending.
    We use 55/45 for more regime switches.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr_sum += max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_hour_from_open_time(open_time):
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
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (1h) indicators
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_sma_1h = calculate_volume_sma(volume, period=20)
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d Choppiness for regime detection
    chop_1d_raw = calculate_choppiness(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    HIGH_CONF_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]):
            continue
        if np.isnan(chop_1d_aligned[i]):
            continue
        if np.isnan(vol_sma_1h[i]) or vol_sma_1h[i] <= 1e-10:
            continue
        
        # Extract UTC hour for session filter
        hour_utc = get_hour_from_open_time(open_time[i])
        
        # === SESSION FILTER (8-20 UTC) — CRITICAL for 1h to reduce trades ===
        in_session = 8 <= hour_utc <= 20
        
        # === TREND BIAS (4h HTF HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION (1d Choppiness Index) ===
        ranging_regime = chop_1d_aligned[i] > 55
        trending_regime = chop_1d_aligned[i] < 45
        
        # === VOLUME CONFIRMATION (relaxed for 1h) ===
        volume_confirmed = volume[i] > 0.8 * vol_sma_1h[i]
        
        # === CRSI SIGNALS (Connors RSI extremes) ===
        crsi_extreme_low = crsi_1h[i] < 20
        crsi_extreme_high = crsi_1h[i] > 80
        crsi_oversold = crsi_1h[i] < 30
        crsi_overbought = crsi_1h[i] > 70
        
        desired_signal = 0.0
        confidence = 0
        
        # === RANGING REGIME (CHOP > 55) — MEAN REVERSION ===
        if ranging_regime:
            # Long: CRSI extreme low + 4h bullish trend + session + volume
            if crsi_extreme_low and trend_4h_bullish:
                confidence = 1
                if in_session:
                    confidence += 1
                if volume_confirmed:
                    confidence += 1
                
                if confidence >= 2:
                    desired_signal = HIGH_CONF_SIZE if confidence >= 3 else BASE_SIZE
            
            # Short: CRSI extreme high + 4h bearish trend + session + volume
            if crsi_extreme_high and trend_4h_bearish:
                confidence = 1
                if in_session:
                    confidence += 1
                if volume_confirmed:
                    confidence += 1
                
                if confidence >= 2:
                    desired_signal = -HIGH_CONF_SIZE if confidence >= 3 else -BASE_SIZE
            
            # Conservative: moderate CRSI + strong trend alignment
            if crsi_oversold and trend_4h_bullish and in_session:
                if desired_signal == 0:
                    desired_signal = BASE_SIZE
            
            if crsi_overbought and trend_4h_bearish and in_session:
                if desired_signal == 0:
                    desired_signal = -BASE_SIZE
        
        # === TRENDING REGIME (CHOP < 45) — TREND FOLLOWING ===
        elif trending_regime:
            # Pullback long: 4h bullish + CRSI recovering from oversold
            if trend_4h_bullish and crsi_oversold and crsi_1h[i] > 15:
                confidence = 1
                if in_session:
                    confidence += 1
                if volume_confirmed:
                    confidence += 1
                
                if confidence >= 2:
                    desired_signal = HIGH_CONF_SIZE if confidence >= 3 else BASE_SIZE
            
            # Pullback short: 4h bearish + CRSI recovering from overbought
            if trend_4h_bearish and crsi_overbought and crsi_1h[i] < 85:
                confidence = 1
                if in_session:
                    confidence += 1
                if volume_confirmed:
                    confidence += 1
                
                if confidence >= 2:
                    desired_signal = -HIGH_CONF_SIZE if confidence >= 3 else -BASE_SIZE
        
        # === NEUTRAL REGIME (45 <= CHOP <= 55) ===
        else:
            # Only trade extreme CRSI with all filters
            if crsi_extreme_low and trend_4h_bullish and in_session and volume_confirmed:
                desired_signal = BASE_SIZE
            
            if crsi_extreme_high and trend_4h_bearish and in_session and volume_confirmed:
                desired_signal = -BASE_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and CRSI not overbought
                if trend_4h_bullish and crsi_1h[i] < 75:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and CRSI not oversold
                if trend_4h_bearish and crsi_1h[i] > 25:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses or CRSI overbought
            if trend_4h_bearish and crsi_1h[i] > 70:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses or CRSI oversold
            if trend_4h_bullish and crsi_1h[i] < 30:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= HIGH_CONF_SIZE:
                desired_signal = HIGH_CONF_SIZE
            else:
                desired_signal = BASE_SIZE
        elif desired_signal < 0:
            if desired_signal <= -HIGH_CONF_SIZE:
                desired_signal = -HIGH_CONF_SIZE
            else:
                desired_signal = -BASE_SIZE
        
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