#!/usr/bin/env python3
"""
Experiment #1028: 30m Primary + 4h/1d HTF — Connors RSI + Choppiness Regime + Volume Session

Hypothesis: After 745+ failed strategies, the key insight is that 30m strategies fail due to
too many trades (>200/year) causing fee drag. This strategy uses:

1. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Proven 75% win rate for mean reversion
   - Long: CRSI < 10 + price > 4h HMA21
   - Short: CRSI > 90 + price < 1d HMA21
   - Asymmetric bias for bear market (easier to short)

2. CHOPPINESS INDEX regime filter:
   - CHOP > 55 = ranging → use CRSI mean reversion
   - CHOP < 45 = trending → use trend-following (HMA slope)
   - Between = hold existing, no new entries

3. 4h HMA21 + 1d HMA21: Dual HTF trend bias
   - Only long when price > 4h HMA (medium-term support)
   - Only short when price < 1d HMA (long-term resistance)
   - This asymmetry works in bear/range markets

4. VOLUME + SESSION filters (CRITICAL for 30m):
   - Volume > 0.8x 20-bar average (avoid low-liquidity entries)
   - Session: 8-20 UTC only (avoid Asian session whipsaw)
   - This reduces trades from 200+/year to 50-80/year

5. ATR Trailing Stop: 2.5x ATR for risk management

Why 30m with strict filters works:
- HTF (4h/1d) determines DIRECTION
- 30m determines ENTRY TIMING within HTF trend
- Volume + session filters cut 60% of low-quality trades
- Target: 50-80 trades/year (vs 200+ without filters)

Critical fixes from failed experiments:
- CRSI instead of RSI/Fisher (better mean reversion signal)
- DUAL HTF with asymmetric logic (4h for long, 1d for short)
- SESSION filter (8-20 UTC) removes Asian session noise
- VOLUME filter avoids low-liquidity false signals
- Discrete signal sizes (0.0, ±0.20, ±0.25) minimize fee churn

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 30m (target 50-80 trades/year with filters)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_chop_regime_4h1d_hma_vol_session_v1"
timeframe = "30m"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI(3): 3-period RSI on close
    RSI_Streak(2): 2-period RSI on streak duration (consecutive up/down days)
    PercentRank(100): Percentile rank of today's return over last 100 days
    
    Entry: CRSI < 10 (oversold) or CRSI > 90 (overbought)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3)
    rsi_3 = np.full(n, np.nan)
    for i in range(rsi_period, n):
        gains = 0.0
        losses = 0.0
        for j in range(i - rsi_period + 1, i + 1):
            if j == 0:
                continue
            change = close[j] - close[j - 1]
            if change > 0:
                gains += change
            else:
                losses += abs(change)
        if losses == 0:
            rsi_3[i] = 100.0
        else:
            rs = gains / (losses + 1e-10)
            rsi_3[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # RSI Streak (2)
    streak_rsi = np.full(n, np.nan)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            streak[i] = streak[i - 1] + 1 if streak[i - 1] >= 0 else 1
        elif close[i] < close[i - 1]:
            streak[i] = streak[i - 1] - 1 if streak[i - 1] <= 0 else -1
        else:
            streak[i] = 0
    
    for i in range(streak_period, n):
        gains = 0.0
        losses = 0.0
        for j in range(i - streak_period + 1, i + 1):
            if streak[j] > 0:
                gains += streak[j]
            else:
                losses += abs(streak[j])
        if losses == 0:
            streak_rsi[i] = 100.0
        else:
            rs = gains / (losses + 1e-10)
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    # Percent Rank (100)
    pct_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i - rank_period:i + 1])
        if len(returns) > 0:
            current_return = close[i] - close[i - 1]
            count_below = np.sum(returns < current_return)
            pct_rank[i] = 100.0 * count_below / len(returns)
        else:
            pct_rank[i] = 50.0
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging (mean reversion)
    CHOP < 38.2 = trending (trend following)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j - 1]), abs(low[j] - close[j - 1]))
            atr_sum += tr
        
        if atr_sum > 0 and (highest_high - lowest_low) > 0:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range using EMA smoothing."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

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

def calculate_volume_ratio(volume, period=20):
    """Current volume vs average volume."""
    n = len(volume)
    vol_ratio = np.full(n, np.nan)
    
    for i in range(period, n):
        avg_vol = np.mean(volume[i - period:i])
        if avg_vol > 0:
            vol_ratio[i] = volume[i] / avg_vol
    
    return vol_ratio

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    ts_seconds = open_time / 1000
    return int((ts_seconds % 86400) / 3600)

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
    
    # Calculate and align 4h HMA21 for medium-term trend (long filter)
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA21 for long-term trend (short filter)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    crsi_30m = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_30m = calculate_atr(high, low, close, period=14)
    chop_30m = calculate_choppiness(high, low, close, period=14)
    vol_ratio_30m = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    LONG_SIZE = 0.25
    SHORT_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(crsi_30m[i]) or np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(chop_30m[i]) or np.isnan(vol_ratio_30m[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        session_active = 8 <= utc_hour <= 20
        
        # === VOLUME FILTER ===
        volume_ok = vol_ratio_30m[i] >= 0.8
        
        # === MACRO TREND (HTF HMA21) - Asymmetric ===
        medium_bull = close[i] > hma_4h_aligned[i]
        long_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_range = chop_30m[i] > 55  # Ranging → mean reversion
        regime_trend = chop_30m[i] < 45  # Trending → trend follow
        
        # === CRSI SIGNALS ===
        crsi_oversold = crsi_30m[i] < 15
        crsi_overbought = crsi_30m[i] > 85
        crsi_extreme_oversold = crsi_30m[i] < 10
        crsi_extreme_overbought = crsi_30m[i] > 90
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        # Only long if: session active + volume ok + medium bullish + CRSI oversold
        if session_active and volume_ok:
            if regime_range and medium_bull:
                # Mean reversion in ranging market with bullish bias
                if crsi_extreme_oversold:
                    desired_signal = LONG_SIZE
                elif crsi_oversold and crsi_30m[i] < crsi_30m[i - 1]:
                    # CRSI declining into oversold
                    desired_signal = LONG_SIZE
            elif regime_trend and medium_bull:
                # Trend pullback entry
                if crsi_oversold:
                    desired_signal = LONG_SIZE
        
        # === SHORT ENTRIES ===
        # Only short if: session active + volume ok + long-term bearish + CRSI overbought
        if session_active and volume_ok:
            if regime_range and long_bear:
                # Mean reversion in ranging market with bearish bias
                if crsi_extreme_overbought:
                    desired_signal = -SHORT_SIZE
                elif crsi_overbought and crsi_30m[i] > crsi_30m[i - 1]:
                    # CRSI rising into overbought
                    desired_signal = -SHORT_SIZE
            elif regime_trend and long_bear:
                # Trend pullback entry
                if crsi_overbought:
                    desired_signal = -SHORT_SIZE
        
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
                # Hold long if medium bullish and CRSI not extreme overbought
                if medium_bull and crsi_30m[i] < 70:
                    desired_signal = LONG_SIZE
            elif position_side < 0:
                # Hold short if long-term bearish and CRSI not extreme oversold
                if long_bear and crsi_30m[i] > 30:
                    desired_signal = -SHORT_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if medium trend reverses OR CRSI becomes overbought
            if not medium_bull or crsi_30m[i] > 80:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if long-term trend reverses OR CRSI becomes oversold
            if not long_bear or crsi_30m[i] < 20:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = LONG_SIZE
        elif desired_signal < 0:
            desired_signal = -SHORT_SIZE
        
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