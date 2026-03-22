#!/usr/bin/env python3
"""
Experiment #558: 30m Primary + 4h/1d HTF — Regime-Adaptive Connors RSI with Session Filter

Hypothesis: After analyzing failed 30m/1h strategies (#548, #550, #555), the pattern is:
- #548 (30m triple confluence): Sharpe=-3.002 — filters TOO STRICT, 0 trades
- #550 (1h session/volume): Sharpe=0.000 — session filter killed ALL entries
- #555 (1h simplified): Sharpe=-0.484 — no regime filter, whipsaw in ranges

For 30m to work with 30-80 trades/year (Rule 10), I need:
1. 1d Choppiness Index for REGIME (range vs trend) — meta-filter
2. 4h HMA(21) for DIRECTION — only trade with HTF trend
3. 30m Connors RSI for ENTRY TIMING — proven 75% win rate mean reversion
4. Session filter (8-20 UTC) — but PERMISSIVE (not strict like #550)
5. Volume > 0.5x avg (not >1.5x which kills trades)
6. Position size: 0.20 base (smaller for 30m per Rule 4/10)

Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Long: CRSI < 15 + price > SMA(200) + HTF uptrend
- Short: CRSI > 85 + price < SMA(200) + HTF downtrend

Why this might beat Sharpe=0.435:
- Regime filter (CHOP) prevents trend strategies in ranges (major failure mode)
- Connors RSI has proven edge in mean reversion (literature-backed)
- 4h HTF direction prevents counter-trend losses
- Session filter captures London/NY overlap liquidity (but not too strict)
- 30m entry timing within 4h/1d trend = optimal per Rule 10

Position sizing: 0.20 base, 0.30 max (discrete per Rule 4)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_crsi_hma_4h1d_session_v1"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI) — proven mean reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Literature: Connors & Alvarez, "ConnorsRSI" (2013) — 75% win rate on extremes.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3) — very short-term momentum
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak — measures consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak (use absolute streak for RSI calculation)
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    
    # Component 3: Percent Rank — where current return ranks vs last 100 bars
    returns = close_s.pct_change().values
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        rank = np.sum(window < current) / len(window)
        percent_rank[i] = rank * 100.0
    
    # Combine components
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP) — regime detection.
    CHOP > 61.8 = range (mean revert)
    CHOP < 38.2 = trend (trend follow)
    
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Choppiness calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) — reduces lag vs EMA."""
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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

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
    
    # Calculate 4h HTF HMA for major trend direction
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, period=50)
    
    # Calculate 1d HTF Choppiness for regime detection
    chop_1d = calculate_choppiness(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        period=14
    )
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi_30m = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    sma_200 = calculate_sma(close, 200)
    
    # Volume moving average for volume filter
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    # Smaller size for 30m vs 4h/1d (Rule 10: lower TF = more trades = smaller size)
    POSITION_SIZE = 0.20
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):  # Start later for SMA(200) and CRSI warmup
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_4h_50_aligned[i]):
            continue
        if np.isnan(chop_1d_aligned[i]) or np.isnan(crsi_30m[i]):
            continue
        if np.isnan(sma_200[i]) or np.isnan(vol_sma_20[i]):
            continue
        
        # === 1D REGIME (Choppiness Index) ===
        # CHOP > 55 = range → use mean reversion (CRSI extremes)
        # CHOP < 45 = trend → use trend following (HTF HMA direction)
        # 45-55 = transition → reduce position size
        chop_value = chop_1d_aligned[i]
        range_regime = chop_value > 55.0
        trend_regime = chop_value < 45.0
        
        # === 4H MAJOR TREND (direction filter) ===
        bull_regime_4h = close[i] > hma_4h_21_aligned[i]
        bear_regime_4h = close[i] < hma_4h_21_aligned[i]
        
        # 4h HMA slope for trend strength
        hma_4h_slope_bull = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_slope_bear = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # === SESSION FILTER (8-20 UTC) — PERMISSIVE ===
        # Extract hour from open_time (milliseconds timestamp)
        # Binance timestamps are in milliseconds
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20  # London/NY overlap + US session
        
        # === VOLUME FILTER — PERMISSIVE ===
        # Volume > 0.5x average (not too strict)
        volume_ok = volume[i] > 0.5 * vol_sma_20[i] if vol_sma_20[i] > 0 else True
        
        # === CONNORS RSI ENTRY (extreme mean reversion) ===
        # Long: CRSI < 15 (oversold) + price > SMA200 (uptrend filter)
        crsi_oversold = crsi_30m[i] < 15.0
        crsi_overbought = crsi_30m[i] > 85.0
        
        # Price vs SMA200 filter
        price_above_sma = close[i] > sma_200[i]
        price_below_sma = close[i] < sma_200[i]
        
        # === ENTRY LOGIC — REGIME ADAPTIVE ===
        new_signal = 0.0
        
        # LONG ENTRY: Multiple confluence paths
        if range_regime:
            # Range regime: CRSI mean reversion + 4h not strongly bearish
            if crsi_oversold and not (bear_regime_4h and hma_4h_slope_bear):
                if in_session and volume_ok:
                    new_signal = POSITION_SIZE
        elif trend_regime:
            # Trend regime: Follow 4h direction on CRSI pullback
            if bull_regime_4h and crsi_oversold and price_above_sma:
                if in_session and volume_ok:
                    new_signal = POSITION_SIZE_MAX if hma_4h_slope_bull else POSITION_SIZE
            elif bear_regime_4h and crsi_overbought and price_below_sma:
                if in_session and volume_ok:
                    new_signal = -POSITION_SIZE_MAX if hma_4h_slope_bear else -POSITION_SIZE
        else:
            # Transition regime: Only strong signals
            if bull_regime_4h and hma_4h_slope_bull and crsi_oversold:
                if in_session and volume_ok:
                    new_signal = POSITION_SIZE
            elif bear_regime_4h and hma_4h_slope_bear and crsi_overbought:
                if in_session and volume_ok:
                    new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
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
        
        # === EXIT CONDITIONS (regime flip or CRSI reversal) ===
        # Exit long on 4h regime flip to strong bear
        if in_position and position_side > 0:
            if bear_regime_4h and hma_4h_slope_bear:
                new_signal = 0.0
            # Exit on CRSI overbought (mean reversion complete)
            if crsi_30m[i] > 70.0:
                new_signal = 0.0
        
        # Exit short on 4h regime flip to strong bull
        if in_position and position_side < 0:
            if bull_regime_4h and hma_4h_slope_bull:
                new_signal = 0.0
            # Exit on CRSI oversold (mean reversion complete)
            if crsi_30m[i] < 30.0:
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
                # Flip position
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