#!/usr/bin/env python3
"""
Experiment #457: 1d Primary + 4h HTF — KAMA Trend + ADX + Choppiness Regime

Hypothesis: After analyzing 456 failed experiments, clear patterns emerge:
1. #446 (12h + 1d HTF) got Sharpe=0.134 — regime switching works on higher TF
2. #447 (1d + 1w HTF) got Sharpe=-0.116 — 1w HTF too slow for 1d entries
3. Current best Sharpe=0.435 uses 1d + 1w but simpler logic

Key insight: 4h HTF aligns better with 1d primary than 1w (less lag).
KAMA (Kaufman Adaptive MA) adapts to volatility better than HMA/EMA.
ADX confirms trend strength before entering trend-following mode.
Choppiness switches between mean-revert (chop) and trend-follow (low chop).

Why this might beat Sharpe=0.435:
- KAMA reduces whipsaw in 2022 crash vs static MA
- 4h HTF provides timely trend confirmation (not as laggy as 1w)
- ADX > 25 filter ensures we only trend-follow when trend exists
- Asymmetric sizing (0.30 long, 0.25 short) protects in bear markets
- Relaxed entry thresholds ensure >=30 trades/symbol on train

Position sizing: 0.25-0.30 (discrete, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 20-50 trades/year on 1d, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_adx_chop_4h_regime_v1"
timeframe = "1d"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency (trend vs noise).
    
    Efficiency Ratio (ER) = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = KAMA_prev + SC * (Close - KAMA_prev)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Net change over er_period
    net_change = np.abs(close_s - close_s.shift(er_period))
    
    # Sum of absolute changes (volatility)
    vol_sum = pd.Series(np.abs(close_s.diff())).rolling(window=er_period, min_periods=er_period).sum()
    
    # Efficiency Ratio (0 to 1)
    er = net_change / (vol_sum + 1e-10)
    er = er.fillna(0)
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]  # Initialize
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    ADX > 25 = trending market
    ADX < 20 = ranging market
    """
    n = len(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values (Wilder's smoothing = EMA with span=period)
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    
    # DI values
    plus_di = 100 * plus_di / (atr + 1e-10)
    minus_di = 100 * minus_di / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

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
    
    Proven 75% win rate in research notes. Best for mean reversion entries.
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
    
    # RSI on absolute streak values
    streak_abs = np.abs(streak)
    streak_s = pd.Series(streak_abs)
    streak_delta = streak_s.diff()
    gain = streak_delta.where(streak_delta > 0, 0.0)
    loss = -streak_delta.where(streak_delta < 0, 0.0)
    avg_gain = gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_loss = loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    rs_streak = avg_gain / (avg_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    
    # Component 3: Percent Rank of daily returns over 100 periods
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
    
    # Combine components
    crsi = (rsi_close + rsi_streak.values + percent_rank.values) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # ATR for each bar
    atr = calculate_atr(high, low, close, period)
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    highest_high = high_s.rolling(window=period, min_periods=period).max().values
    lowest_low = low_s.rolling(window=period, min_periods=period).min().values
    
    # Choppiness calculation
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    chop = 100.0 * np.log10(atr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HTF indicators (trend confirmation)
    kama_4h_20 = calculate_kama(df_4h['close'].values, er_period=10, fast_period=2, slow_period=30)
    adx_4h = calculate_adx(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, period=14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_4h_20_aligned = align_htf_to_ltf(prices, df_4h, kama_4h_20)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_1d_20 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_1d_50 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    # Recalculate KAMA-50 with different parameters
    kama_1d_50 = calculate_kama(close, er_period=14, fast_period=2, slow_period=30)
    adx_1d = calculate_adx(high, low, close, period=14)
    crsi_1d = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_1d = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_price = 0.0
    lowest_price = 0.0
    entry_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(kama_4h_20_aligned[i]) or np.isnan(adx_4h_aligned[i]):
            continue
        if np.isnan(kama_1d_20[i]) or np.isnan(kama_1d_50[i]):
            continue
        if np.isnan(adx_1d[i]) or np.isnan(crsi_1d[i]) or np.isnan(chop_1d[i]) or np.isnan(sma_200[i]):
            continue
        
        # === 4H HTF TREND (primary direction filter) ===
        # Price above 4h KAMA = bull bias (favor longs)
        # Price below 4h KAMA = bear bias (favor shorts)
        bull_htf = close[i] > kama_4h_20_aligned[i]
        bear_htf = close[i] < kama_4h_20_aligned[i]
        
        # 4h ADX confirms trend strength
        trend_4h = adx_4h_aligned[i] > 22.0  # relaxed from 25 for more trades
        
        # === CHOPPINESS REGIME DETECTION ===
        # CHOP > 55 = ranging (mean reversion)
        # CHOP < 45 = trending (trend follow)
        is_ranging = chop_1d[i] > 55.0
        is_trending = chop_1d[i] < 45.0
        
        # === 1D LOCAL TREND (KAMA crossover) ===
        kama_bullish = kama_1d_20[i] > kama_1d_50[i]
        kama_bearish = kama_1d_20[i] < kama_1d_50[i]
        
        # 1d ADX for local trend strength
        trend_1d = adx_1d[i] > 20.0
        
        # === CONNORS RSI SIGNALS (mean reversion) ===
        crsi_oversold = crsi_1d[i] < 25.0  # relaxed from 20 for more trades
        crsi_overbought = crsi_1d[i] > 75.0  # relaxed from 80 for more trades
        crsi_extreme_oversold = crsi_1d[i] < 15.0
        crsi_extreme_overbought = crsi_1d[i] > 85.0
        
        # === SMA200 FILTER (long-term trend) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC — REGIME ADAPTIVE (SIMPLIFIED FOR MORE TRADES) ===
        new_signal = 0.0
        
        # LONG ENTRIES
        if bull_htf or above_sma200:
            # Ranging market: mean reversion on CRSI oversold
            if is_ranging and crsi_oversold:
                new_signal = LONG_SIZE
            # Trending market: KAMA bullish + ADX confirmation
            elif is_trending and kama_bullish and trend_1d:
                new_signal = LONG_SIZE
            # 4h trend + 1d pullback (CRSI not overbought)
            elif bull_htf and kama_bullish and crsi_1d[i] < 60.0:
                new_signal = LONG_SIZE * 0.8
            # CRSI extreme oversold (works in any regime)
            elif crsi_extreme_oversold:
                new_signal = LONG_SIZE * 0.7
        
        # SHORT ENTRIES
        if bear_htf or below_sma200:
            # Ranging market: mean reversion on CRSI overbought
            if is_ranging and crsi_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # Trending market: KAMA bearish + ADX confirmation
            elif is_trending and kama_bearish and trend_1d:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE
            # 4h trend + 1d bounce (CRSI not oversold)
            elif bear_htf and kama_bearish and crsi_1d[i] > 40.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.8
            # CRSI extreme overbought (works in any regime)
            elif crsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.7
        
        # === FREQUENCY BOOST (ensure >=30 trades/symbol on train) ===
        # If no position and weak signal, enter on simpler conditions
        if not in_position and new_signal == 0.0:
            # Long: CRSI < 30 + KAMA bullish (simpler entry)
            if bull_htf and kama_bullish and crsi_1d[i] < 30.0:
                new_signal = LONG_SIZE * 0.5
            # Short: CRSI > 70 + KAMA bearish (simpler entry)
            elif bear_htf and kama_bearish and crsi_1d[i] > 70.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.5
            # KAMA crossover alone (trend following)
            elif kama_bullish and adx_1d[i] > 18.0:
                new_signal = LONG_SIZE * 0.4
            elif kama_bearish and adx_1d[i] > 18.0:
                if new_signal == 0.0:
                    new_signal = -SHORT_SIZE * 0.4
        
        # === STOPLOSS CHECK (BEFORE exit logic - CRITICAL) ===
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
        # CRSI extreme exit (take profit on exhaustion)
        if in_position and position_side > 0 and crsi_1d[i] > 85.0:
            new_signal = 0.0
        if in_position and position_side < 0 and crsi_1d[i] < 15.0:
            new_signal = 0.0
        
        # Trend reversal exit (4h HTF flip)
        if in_position and position_side > 0 and bear_htf and kama_bearish:
            new_signal = 0.0
        if in_position and position_side < 0 and bull_htf and kama_bullish:
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
                # Position flip
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