#!/usr/bin/env python3
"""
Experiment #676: 12h Primary + 1d HTF — Dual Regime (Trend/Mean-Revert) + Connors RSI

Hypothesis: 12h timeframe with daily HTF filter provides optimal balance between
signal quality and trade frequency. Key innovations:
1. Connors RSI (CRSI) for mean-reversion entries — proven 75% win rate in ranges
2. ADX regime switch — ADX>25=trend-follow, ADX<20=mean-revert, hysteresis at 22
3. Donchian breakout for trend entries — captures sustained moves
4. 1d HMA for macro bias — prevents counter-trend trades
5. LOOSE entry thresholds (CRSI<20/>80, not <10/>90) to ensure trade generation
6. Position size 0.25-0.30 with 2.5x ATR trailing stop

Why this should work where #673 failed:
- #673 used funding rate which may not align with price data properly
- This uses pure price action indicators (no external data dependencies)
- 12h TF = ~25-45 trades/year (low fee drag, sufficient trade count)
- Dual regime adapts to market conditions (trend vs range)
- Connors RSI is proven edge for crypto mean-reversion

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_crsi_donchian_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) — composite mean-reversion indicator.
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    Long: CRSI < 20 | Short: CRSI > 80
    Proven 75% win rate for crypto mean-reversion.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period:
        return crsi
    
    # RSI(close, 3)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(np.concatenate([[0], gain])).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(np.concatenate([[0], loss])).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_close = 100 - (100 / (1 + rs))
    rsi_close = np.clip(rsi_close, 0, 100)
    
    # RSI(streak, 2) — streak = consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_abs = np.abs(streak)
    streak_gain = np.where(streak > 0, streak_abs, 0)
    streak_loss = np.where(streak < 0, streak_abs, 0)
    
    avg_streak_gain = pd.Series(streak_gain).rolling(window=streak_period, min_periods=streak_period).mean().values
    avg_streak_loss = pd.Series(streak_loss).rolling(window=streak_period, min_periods=streak_period).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100 - (100 / (1 + rs_streak))
    rsi_streak = np.clip(rsi_streak, 0, 100)
    
    # PercentRank(100) — where does current close rank vs last 100?
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i-rank_period:i+1]
        rank = np.sum(window[:-1] < close[i])
        percent_rank[i] = rank / (rank_period - 1) * 100
    
    # Combine into CRSI
    with np.errstate(invalid='ignore'):
        crsi = (rsi_close + rsi_streak + percent_rank) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_adx(high, low, close, period=14):
    """Average Directional Index — measures trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 3:
        return adx
    
    # True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
        
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * plus_di / (atr + 1e-10)
        minus_di = 100 * minus_di / (atr + 1e-10)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_hma(close, period=21):
    """Hull Moving Average — smoother than EMA, less lag."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel — breakout levels."""
    n = len(close := high)  # use high for length
    
    donchian_upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    return donchian_upper, donchian_lower, donchian_mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (12h) indicators
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=50)
    adx_12h = calculate_adx(high, low, close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    
    # Calculate and align HTF (1d) indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # ADX on daily for macro trend strength
    adx_1d_raw = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30
    SIZE_SHORT = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Regime hysteresis tracking
    prev_regime = 0  # 0=unknown, 1=trend, 2=range
    
    for i in range(100, n):  # Start after warmup period
        # Skip if indicators not ready
        if np.isnan(crsi_12h[i]) or np.isnan(adx_12h[i]):
            continue
        if np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === REGIME DETECTION (ADX with hysteresis) ===
        # Use 12h ADX for primary regime, 1d ADX for confirmation
        adx_12h_val = adx_12h[i]
        adx_1d_val = adx_1d_aligned[i]
        
        # Hysteresis: enter trend at 25, exit at 20; enter range at 20, exit at 25
        if prev_regime == 1:  # Was in trend regime
            is_trend_regime = adx_12h_val > 20  # Exit trend only below 20
            is_range_regime = not is_trend_regime
        elif prev_regime == 2:  # Was in range regime
            is_range_regime = adx_12h_val < 25  # Exit range only above 25
            is_trend_regime = not is_range_regime
        else:  # Unknown initial regime
            is_trend_regime = adx_12h_val > 25
            is_range_regime = adx_12h_val < 20
        
        # Update regime tracking
        if is_trend_regime:
            prev_regime = 1
        elif is_range_regime:
            prev_regime = 2
        
        # === DAILY MACRO BIAS (1d HMA + ADX) ===
        daily_bullish = close[i] > hma_1d_aligned[i]
        daily_bearish = close[i] < hma_1d_aligned[i]
        daily_trending = adx_1d_val > 20
        
        # === 12h TREND DIRECTION ===
        price_above_hma = close[i] > hma_1d_aligned[i]
        
        # === CRSI SIGNALS (LOOSE thresholds for trade generation) ===
        crsi_oversold = crsi_12h[i] < 25  # Was <20, loosened to <25
        crsi_overbought = crsi_12h[i] > 75  # Was >80, loosened to >75
        crsi_extreme_oversold = crsi_12h[i] < 15
        crsi_extreme_overbought = crsi_12h[i] > 85
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_down = close[i] < donchian_lower[i-1] if i > 0 else False
        
        desired_signal = 0.0
        
        # === REGIME 1: TRENDING (ADX > 20-25) — Trend Follow ===
        if is_trend_regime:
            # Long: Daily bullish + Donchian breakout OR pullback
            if daily_bullish and daily_trending:
                if donchian_breakout_up:
                    desired_signal = SIZE_LONG
                elif crsi_oversold and price_above_hma:
                    # Pullback entry in uptrend
                    desired_signal = SIZE_LONG * 0.7
            
            # Short: Daily bearish + Donchian breakdown OR rally
            elif daily_bearish and daily_trending:
                if donchian_breakout_down:
                    desired_signal = -SIZE_SHORT
                elif crsi_overbought and not price_above_hma:
                    # Rally entry in downtrend
                    desired_signal = -SIZE_SHORT * 0.7
        
        # === REGIME 2: RANGING (ADX < 20-25) — Mean Reversion ===
        elif is_range_regime:
            # Long: CRSI oversold + near Donchian lower
            if crsi_extreme_oversold or (crsi_oversold and close[i] < donchian_mid[i]):
                desired_signal = SIZE_LONG
            
            # Short: CRSI overbought + near Donchian upper
            if crsi_extreme_overbought or (crsi_overbought and close[i] > donchian_mid[i]):
                desired_signal = -SIZE_SHORT
        
        # === REGIME 3: TRANSITION — Use CRSI extremes only ===
        else:
            if crsi_extreme_oversold:
                desired_signal = SIZE_LONG * 0.5
            elif crsi_extreme_overbought:
                desired_signal = -SIZE_SHORT * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if regime unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if daily still bullish OR CRSI not extreme overbought
                if daily_bullish or crsi_12h[i] < 80:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if daily still bearish OR CRSI not extreme oversold
                if daily_bearish or crsi_12h[i] > 20:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            # If same side, update trailing stop levels
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