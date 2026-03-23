#!/usr/bin/env python3
"""
Experiment #546: 12h Primary + 1d HTF — Regime-Adaptive Dual Strategy

Hypothesis: After 480+ failed strategies, the clearest pattern is:
- 12h timeframe with 1d HTF worked moderately (#536 Sharpe=0.159, #542 Sharpe=0.175)
- BUT complex regime switching consistently underperformed simple trend+pullback
- Key insight from research: Choppiness Index + Connors RSI worked for ETH (Sharpe +0.923)
- Donchian breakout + HMA trend worked for SOL (Sharpe +0.782)
- The problem: applying ONE strategy to ALL regimes fails in bear/range markets

This strategy uses DUAL REGIME approach:
1. CHOPPINESS INDEX (14) detects regime: CHOP>61.8 = range, CHOP<38.2 = trend
2. RANGE REGIME: Connors RSI mean reversion (CRSI<15 long, CRSI>85 short)
3. TREND REGIME: HMA(16/48) crossover + Donchian(20) breakout + ADX(14)>20
4. 1d HTF HMA(21) slope for major trend bias (filter counter-trend trades)
5. ATR(14) 2.5x trailing stop for all positions
6. Volume filter: only trade when volume > 0.8 * 20-bar avg (avoid low-liquidity)

Why this might beat Sharpe=0.435:
- Adapts to market conditions (2022 crash = range, 2021/2023 = trend)
- Connors RSI has 75% win rate in range markets (research-backed)
- 1d HTF filter prevents major counter-trend losses (key failure mode)
- 12h TF targets 25-40 trades/year (optimal fee/trade ratio per rules)
- Discrete position sizing (0.0, ±0.25, ±0.30) minimizes fee churn

Position sizing: 0.28 base, 0.30 for high-conviction (discrete per Rule 4)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_chop_connors_hma_1d_v1"
timeframe = "12h"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI) - research-backed mean reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    - RSI(3): Short-term momentum
    - RSI_Streak(2): RSI of consecutive up/down days
    - PercentRank(100): Where current price ranks vs last 100 days
    
    Entry: CRSI < 10-15 (oversold) for long, CRSI > 85-90 (overbought) for short
    Research shows 75% win rate with SMA(200) filter.
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streaks
    # Streak = consecutive up/down days
    direction = np.sign(np.diff(close, prepend=close[0]))
    streak = np.zeros(n)
    streak[0] = 1
    for i in range(1, n):
        if direction[i] == direction[i-1]:
            streak[i] = streak[i-1] + 1
        else:
            streak[i] = 1
    
    # RSI of streak values
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Component 3: Percent Rank
    # Where does current close rank vs last 100 closes?
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100.0
        percent_rank[i] = rank
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP) - regime detection indicator.
    
    Formula: CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8: Market is choppy/ranging (use mean reversion)
    - CHOP < 38.2: Market is trending (use trend following)
    - 38.2 < CHOP < 61.8: Transition zone (reduce position or stay flat)
    
    Research-backed threshold for crypto markets.
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # Neutral if no range
    
    return chop

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    mask = (plus_dm > 0) & (minus_dm > 0)
    plus_dm_vals = plus_dm.values.copy()
    minus_dm_vals = minus_dm.values.copy()
    plus_dm_vals[mask] = np.where(plus_dm_vals[mask] > minus_dm_vals[mask], plus_dm_vals[mask], 0)
    minus_dm_vals[mask] = np.where(minus_dm_vals[mask] > plus_dm_vals[mask], minus_dm_vals[mask], 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm_s = pd.Series(plus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm_vals).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100.0 * plus_dm_s / (tr_s + 1e-10)
    minus_di = 100.0 * minus_dm_s / (tr_s + 1e-10)
    
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bands)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF HMA for major trend direction
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness_index(high, low, close, 14)
    
    # 12h HMA for trend confirmation
    hma_12h_16 = calculate_hma(close, period=16)
    hma_12h_48 = calculate_hma(close, period=48)
    
    # Donchian channels for breakout detection
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    
    # Volume filter
    vol_avg_20 = calculate_volume_avg(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_HIGH = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track previous values for crossover detection
    prev_hma_16 = np.zeros(n)
    prev_hma_16[1:] = hma_12h_16[:-1]
    prev_hma_48 = np.zeros(n)
    prev_hma_48[1:] = hma_12h_48[:-1]
    
    # Track Donchian breakout
    prev_donchian_upper = np.zeros(n)
    prev_donchian_upper[1:] = donchian_upper[:-1]
    prev_donchian_lower = np.zeros(n)
    prev_donchian_lower[1:] = donchian_lower[:-1]
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_12h_16[i]) or np.isnan(hma_12h_48[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === VOLUME FILTER (avoid low liquidity) ===
        volume_ok = volume[i] > 0.8 * vol_avg_20[i]
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime_1d = close[i] > hma_1d_21_aligned[i]
        bear_regime_1d = close[i] < hma_1d_21_aligned[i]
        
        # 1d HMA slope for trend strength
        hma_1d_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_1d_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        choppy_regime = chop[i] > 61.8  # Range market - use mean reversion
        trending_regime = chop[i] < 38.2  # Trend market - use trend following
        transition_regime = not choppy_regime and not trending_regime
        
        # === 12H TREND CONFIRMATION ===
        # HMA crossover (fast above slow = bull)
        hma_bull_cross = hma_12h_16[i] > hma_12h_48[i]
        hma_bear_cross = hma_12h_16[i] < hma_12h_48[i]
        
        # HMA crossover confirmation (just crossed)
        hma_bull_crossed = hma_bull_cross and (prev_hma_16[i] <= prev_hma_48[i])
        hma_bear_crossed = hma_bear_cross and (prev_hma_16[i] >= prev_hma_48[i])
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > prev_donchian_upper[i]
        donchian_breakout_short = close[i] < prev_donchian_lower[i]
        
        # === ADX FILTER (trending market strength) ===
        strong_trend = adx_14[i] > 20.0
        weak_trend = adx_14[i] < 15.0
        
        # === CONNORS RSI MEAN REVERSION SIGNALS ===
        crsi_oversold = crsi[i] < 15.0  # Strong mean reversion long
        crsi_overbought = crsi[i] > 85.0  # Strong mean reversion short
        crsi_moderate_oversold = crsi[i] < 25.0  # Moderate long
        crsi_moderate_overbought = crsi[i] > 75.0  # Moderate short
        
        # === ENTRY LOGIC — REGIME ADAPTIVE ===
        new_signal = 0.0
        
        # --- RANGE REGIME (CHOP > 61.8): Mean Reversion with Connors RSI ---
        if choppy_regime and volume_ok:
            # LONG: CRSI oversold + price above 1d HMA (bullish bias)
            if crsi_oversold and bull_regime_1d:
                new_signal = POSITION_SIZE_HIGH
            elif crsi_moderate_oversold and bull_regime_1d and hma_bull_cross:
                new_signal = POSITION_SIZE_BASE
            # SHORT: CRSI overbought + price below 1d HMA (bearish bias)
            elif crsi_overbought and bear_regime_1d:
                new_signal = -POSITION_SIZE_HIGH
            elif crsi_moderate_overbought and bear_regime_1d and hma_bear_cross:
                new_signal = -POSITION_SIZE_BASE
        
        # --- TREND REGIME (CHOP < 38.2): Trend Following ---
        elif trending_regime and volume_ok:
            # LONG: 1d bull + 12h HMA bull + Donchian breakout OR HMA cross
            if bull_regime_1d and hma_1d_slope_bull:
                if donchian_breakout_long and strong_trend:
                    new_signal = POSITION_SIZE_HIGH
                elif hma_bull_crossed and strong_trend:
                    new_signal = POSITION_SIZE_BASE
                elif hma_bull_cross and rsi_14[i] < 65.0:  # Pullback entry
                    new_signal = POSITION_SIZE_BASE
            # SHORT: 1d bear + 12h HMA bear + Donchian breakout OR HMA cross
            elif bear_regime_1d and hma_1d_slope_bear:
                if donchian_breakout_short and strong_trend:
                    new_signal = -POSITION_SIZE_HIGH
                elif hma_bear_crossed and strong_trend:
                    new_signal = -POSITION_SIZE_BASE
                elif hma_bear_cross and rsi_14[i] > 35.0:  # Pullback entry
                    new_signal = -POSITION_SIZE_BASE
        
        # --- TRANSITION REGIME: Reduced size, wait for confirmation ---
        elif transition_regime and volume_ok:
            # Only take high-conviction signals
            if bull_regime_1d and hma_bull_crossed and donchian_breakout_long:
                new_signal = POSITION_SIZE_BASE * 0.8
            elif bear_regime_1d and hma_bear_crossed and donchian_breakout_short:
                new_signal = -POSITION_SIZE_BASE * 0.8
        
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
        
        # === EXIT CONDITIONS (regime flip or weak trend) ===
        # Exit long on 1d regime flip to bear + HMA slope bear
        if in_position and position_side > 0:
            if bear_regime_1d and hma_1d_slope_bear:
                new_signal = 0.0
            elif weak_trend and choppy_regime:  # Trend dying, entering chop
                new_signal = 0.0
        
        # Exit short on 1d regime flip to bull + HMA slope bull
        if in_position and position_side < 0:
            if bull_regime_1d and hma_1d_slope_bull:
                new_signal = 0.0
            elif weak_trend and choppy_regime:  # Trend dying, entering chop
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