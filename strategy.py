#!/usr/bin/env python3
"""
Experiment #468: 30m Primary + 4h/1d HTF — Connors RSI + Choppiness + Vol Spike

Hypothesis: After 467 experiments, clear pattern for lower TF success:
1. 30m needs 4h trend direction (1d too slow, causes lag entries)
2. Connors RSI (CRSI) proven 75% win rate - use wider thresholds (15/85) for MORE trades
3. Choppiness Index regime: CHOP>55=range(mean revert), CHOP<45=trend(follow)
4. Vol spike filter: ATR(7)/ATR(30)>1.8 = panic/reversal opportunity
5. NO session filter - previous 30m strategies got 0 trades due to over-filtering
6. Size=0.20-0.25 (smaller for 30m to reduce fee drag)

Why this might beat current best (Sharpe=0.435):
- 4h HMA gives faster trend signal than 1d for 30m entries
- CRSI wider thresholds (15/85 vs 10/90) = 2-3x more trades
- Vol spike filter catches panic reversals (best edge in bear markets)
- Fewer conflicting filters = guaranteed trades (fixes #458, #465 zero-trade bug)
- 30m TF with HTF direction = HTF trade count with 30m precision

Position sizing: 0.20-0.25 (discrete, max 0.30 for 30m)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 40-80 trades/year on 30m, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_crsi_chop_volspike_4h_v1"
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
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Proven 75% win rate. Use wider thresholds (15/85) for more trades.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI on streak length
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_abs = np.abs(streak)
    streak_s = pd.Series(streak_abs)
    streak_delta = streak_s.diff()
    gain = streak_delta.where(streak_delta > 0, 0.0)
    loss = -streak_delta.where(streak_delta < 0, 0.0)
    avg_gain = gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_loss = loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    rs_streak = avg_gain / (avg_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    
    # Component 3: Percent Rank of returns over 100 periods
    returns = close_s.pct_change()
    percent_rank = pd.Series(np.zeros(n))
    for i in range(rank_period, n):
        window = returns.iloc[i-rank_period:i]
        current = returns.iloc[i]
        if np.isnan(current):
            percent_rank.iloc[i] = 50.0
        else:
            rank = (window < current).sum()
            percent_rank.iloc[i] = (rank / rank_period) * 100.0
    
    crsi = (rsi_close + rsi_streak.values + percent_rank.values) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    Use 55/45 thresholds for more regime switches.
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    atr = calculate_atr(high, low, close, period)
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    highest_high = high_s.rolling(window=period, min_periods=period).max().values
    lowest_low = low_s.rolling(window=period, min_periods=period).min().values
    
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    chop = 100.0 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    crsi_30m = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_30m = calculate_choppiness(high, low, close, period=14)
    
    # Volume MA for filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, smaller for 30m)
    LONG_SIZE = 0.25
    SHORT_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_50_aligned[i]):
            continue
        if np.isnan(crsi_30m[i]) or np.isnan(chop_30m[i]):
            continue
        if np.isnan(vol_ma[i]) or vol_ma[i] == 0:
            continue
        
        # === 4H MAJOR TREND (primary direction filter) ===
        hma_4h_bullish = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_bearish = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # Price relative to 4h HMA
        price_above_4h = close[i] > hma_4h_21_aligned[i]
        price_below_4h = close[i] < hma_4h_21_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        is_ranging = chop_30m[i] > 55.0
        is_trending = chop_30m[i] < 45.0
        
        # === CONNORS RSI SIGNALS (wider thresholds for MORE trades) ===
        crsi_oversold = crsi_30m[i] < 25.0  # relaxed from 20
        crsi_overbought = crsi_30m[i] > 75.0  # relaxed from 80
        crsi_extreme_oversold = crsi_30m[i] < 15.0  # relaxed from 10
        crsi_extreme_overbought = crsi_30m[i] > 85.0  # relaxed from 90
        
        # === VOL SPIKE FILTER (panic/reversal opportunity) ===
        vol_spike = (atr_7[i] / atr_30[i]) > 1.8  # vol expansion
        vol_normal = (atr_7[i] / atr_30[i]) < 1.2  # vol contraction
        
        # === VOLUME FILTER (not too strict) ===
        vol_ok = volume[i] > 0.6 * vol_ma[i]  # relaxed from 0.8
        
        # === ENTRY LOGIC — DESIGNED FOR TRADE FREQUENCY ===
        new_signal = 0.0
        
        # LONG ENTRIES (multiple paths to ensure trades)
        if hma_4h_bullish or price_above_4h:
            # Path 1: Ranging + CRSI oversold (mean reversion)
            if is_ranging and crsi_oversold and vol_ok:
                new_signal = LONG_SIZE
            # Path 2: Trending + pullback (trend follow)
            elif is_trending and hma_4h_bullish and crsi_30m[i] < 40.0 and vol_ok:
                new_signal = LONG_SIZE
            # Path 3: Extreme CRSI (works any regime)
            elif crsi_extreme_oversold and vol_ok:
                new_signal = LONG_SIZE
            # Path 4: Vol spike + oversold (panic reversal)
            elif vol_spike and crsi_30m[i] < 30.0:
                new_signal = LONG_SIZE * 0.8
            # Path 5: Simple HMA + CRSI (fallback for more trades)
            elif hma_4h_bullish and crsi_30m[i] < 35.0:
                new_signal = LONG_SIZE * 0.7
        
        # SHORT ENTRIES (multiple paths)
        if hma_4h_bearish or price_below_4h:
            # Path 1: Ranging + CRSI overbought
            if is_ranging and crsi_overbought and vol_ok:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Path 2: Trending + bounce
            elif is_trending and hma_4h_bearish and crsi_30m[i] > 60.0 and vol_ok:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Path 3: Extreme CRSI
            elif crsi_extreme_overbought and vol_ok:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Path 4: Vol spike + overbought
            elif vol_spike and crsi_30m[i] > 70.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
            # Path 5: Simple HMA + CRSI
            elif hma_4h_bearish and crsi_30m[i] > 65.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.7
        
        # === STOPLOSS CHECK (BEFORE exit logic) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_price = max(highest_price, close[i])
            stop_price = highest_price - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_price == 0.0:
                lowest_price = close[i]
            else:
                lowest_price = min(lowest_price, close[i])
            stop_price = lowest_price + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS ===
        # CRSI extreme exit (take profit)
        if in_position and position_side > 0 and crsi_30m[i] > 80.0:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi_30m[i] < 20.0:
            new_signal = 0.0
        
        # 4H trend reversal exit
        if in_position and position_side > 0 and hma_4h_bearish and price_below_4h:
            new_signal = 0.0
        if in_position and position_side < 0 and hma_4h_bullish and price_above_4h:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals