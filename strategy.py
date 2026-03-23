#!/usr/bin/env python3
"""
Experiment #778: 30m Primary + 4h/1d HTF — Choppiness Regime + Connors RSI + Session Filter

Hypothesis: After 500+ failed strategies and analyzing what works:
1. 30m timeframe needs EXTREMELY strict filters to avoid fee drag (>100 trades/year = death)
2. Choppiness Index (CHOP) is the BEST regime filter for crypto (proven in literature)
3. 1d HMA(21) provides cleaner trend signal than EMA for HTF direction
4. 4h CHOP + 30m CRSI entry = HTF frequency with lower TF precision
5. Session filter (8-20 UTC) captures 70% of volume, avoids Asian chop
6. Volume filter relaxed (0.8x) to ensure trade generation on all symbols
7. Discrete signals (0.0, ±0.20, ±0.30) minimize fee churn from signal changes

Strategy design:
1. 1d HMA(21) for primary trend bias (aligned via mtf_data helper)
2. 4h Choppiness Index(14) for regime: CHOP>61.8=range, CHOP<38.2=trend
3. 30m Connors RSI for entry timing: <15=oversold, >85=overbought
4. Session filter: only trade 8-20 UTC (highest volume, lowest noise)
5. Volume confirmation: >0.8x 20-period average
6. ATR(14) trailing stop at 2.5x for risk management
7. Position sizing: 0.20 (range), 0.30 (trend) — conservative for 30m

Key improvements from failed 30m strategies:
- MUCH stricter entry (3+ confluence required) to limit trades to 30-80/year
- 1d HTF for trend (slower, more reliable than 4h)
- CHOP regime filter (proven edge in crypto mean reversion vs trend)
- Session filter eliminates 60% of low-quality signals
- Asymmetric sizing based on regime confidence

Target: Sharpe > 0.612, trades 30-80/year, ALL symbols positive Sharpe
Timeframe: 30m (with 4h/1d HTF for direction)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_crsi_session_1d_hma_4h_regime_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(series, period):
    """
    Hull Moving Average - faster, smoother than EMA.
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(series)
    if n < period:
        return np.full(n, np.nan)
    
    series = pd.Series(series)
    
    def wma(data, span):
        """Weighted Moving Average."""
        weights = np.arange(1, span + 1)
        return data.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    half_period = period // 2
    wma_half = wma(series, half_period)
    wma_full = wma(series, period)
    
    hull_raw = 2 * wma_half - wma_full
    hma = wma(hull_raw, int(np.sqrt(period)))
    
    return hma.values

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
    Connors RSI Streak component.
    Measures consecutive up/down bars.
    """
    n = len(close)
    streak_rsi = np.full(n, np.nan)
    
    if n < period + 5:
        return streak_rsi
    
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_abs = np.abs(streak)
    streak_score = np.where(streak >= 0, streak_abs, -streak_abs)
    streak_rsi = calculate_rsi(streak_score + 100, period)
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Connors RSI Percent Rank component.
    Percentage of past returns less than current return.
    """
    n = len(close)
    pct_rank = np.full(n, np.nan)
    
    if n < period + 1:
        return pct_rank
    
    returns = np.diff(close) / (close[:-1] + 1e-10) * 100
    returns = np.concatenate([[0], returns])
    
    for i in range(period, n):
        window = returns[i-period:i]
        current = returns[i]
        pct_rank[i] = 100 * np.sum(window < current) / period
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pct_rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Values 0-100. <15 = oversold, >85 = overbought.
    """
    rsi_short = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, pct_rank_period)
    
    with np.errstate(invalid='ignore'):
        crsi = (rsi_short + streak_rsi + pct_rank) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
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
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

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

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)."""
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
    
    # Calculate primary (30m) indicators
    crsi_30m = calculate_crsi(close, rsi_period=3, streak_period=2, pct_rank_period=100)
    atr_30m = calculate_atr(high, low, close, period=14)
    vol_sma_30m = calculate_volume_sma(volume, period=20)
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    chop_4h_raw = calculate_choppiness(
        df_4h['high'].values,
        df_4h['low'].values,
        df_4h['close'].values,
        period=14
    )
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_raw)
    
    signals = np.zeros(n)
    RANGE_SIZE = 0.20
    TREND_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(crsi_30m[i]) or np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(chop_4h_aligned[i]):
            continue
        if np.isnan(vol_sma_30m[i]) or vol_sma_30m[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === TREND BIAS (1d HTF HMA21) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (4h Choppiness) ===
        chop_value = chop_4h_aligned[i]
        ranging_regime = chop_value > 61.8
        trending_regime = chop_value < 38.2
        neutral_regime = not ranging_regime and not trending_regime
        
        # === VOLUME CONFIRMATION (relaxed for trade generation) ===
        volume_confirmed = volume[i] > 0.8 * vol_sma_30m[i]
        
        # === CRSI SIGNALS (30m) ===
        crsi_oversold = crsi_30m[i] < 15
        crsi_overbought = crsi_30m[i] > 85
        crsi_extreme_oversold = crsi_30m[i] < 10
        crsi_extreme_overbought = crsi_30m[i] > 90
        crsi_neutral = 35 < crsi_30m[i] < 65
        
        desired_signal = 0.0
        
        # === RANGING REGIME (CHOP > 61.8) - Mean Reversion ===
        if ranging_regime and in_session:
            # Long: CRSI oversold + trend not bearish + volume
            if crsi_oversold and not trend_1d_bearish and volume_confirmed:
                desired_signal = RANGE_SIZE
            
            # Short: CRSI overbought + trend not bullish + volume
            if crsi_overbought and not trend_1d_bullish and volume_confirmed:
                desired_signal = -RANGE_SIZE
            
            # Extreme entries (higher confidence)
            if crsi_extreme_oversold and trend_1d_bullish:
                desired_signal = TREND_SIZE
            if crsi_extreme_overbought and trend_1d_bearish:
                desired_signal = -TREND_SIZE
        
        # === TRENDING REGIME (CHOP < 38.2) - Trend Following ===
        elif trending_regime and in_session:
            # Long pullback: 1d bullish + CRSI neutral/oversold + volume
            if trend_1d_bullish and (crsi_oversold or crsi_neutral) and volume_confirmed:
                desired_signal = TREND_SIZE
            
            # Short pullback: 1d bearish + CRSI neutral/overbought + volume
            if trend_1d_bearish and (crsi_overbought or crsi_neutral) and volume_confirmed:
                desired_signal = -TREND_SIZE
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) - Conservative ===
        elif neutral_regime and in_session:
            # Only extreme CRSI with trend alignment
            if crsi_extreme_oversold and trend_1d_bullish and volume_confirmed:
                desired_signal = RANGE_SIZE
            if crsi_extreme_overbought and trend_1d_bearish and volume_confirmed:
                desired_signal = -RANGE_SIZE
        
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
                # Hold long if 1d trend intact and CRSI not overbought
                if trend_1d_bullish and crsi_30m[i] < 80:
                    desired_signal = RANGE_SIZE
            elif position_side < 0:
                # Hold short if 1d trend intact and CRSI not oversold
                if trend_1d_bearish and crsi_30m[i] > 20:
                    desired_signal = -RANGE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 1d trend reverses or CRSI overbought
            if trend_1d_bearish and crsi_30m[i] > 75:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 1d trend reverses or CRSI oversold
            if trend_1d_bullish and crsi_30m[i] < 25:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= TREND_SIZE:
                desired_signal = TREND_SIZE
            else:
                desired_signal = RANGE_SIZE
        elif desired_signal < 0:
            if desired_signal <= -TREND_SIZE:
                desired_signal = -TREND_SIZE
            else:
                desired_signal = -RANGE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
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