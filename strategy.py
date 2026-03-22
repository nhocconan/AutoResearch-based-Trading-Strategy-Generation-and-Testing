#!/usr/bin/env python3
"""
Experiment #001: 4h Regime-Adaptive Strategy with Choppiness Index + Connors RSI

Hypothesis: Previous strategies failed because they used ONE approach (trend OR mean revert)
for ALL market conditions. This strategy ADAPTS to regime:

1. CHOPPINESS INDEX (CHOP) detects regime:
   - CHOP > 61.8 = RANGING market → use mean reversion (Connors RSI)
   - CHOP < 38.2 = TRENDING market → use trend following (KAMA)
   - 38.2 <= CHOP <= 61.8 = transition → stay flat or reduce size

2. CONNORS RSI (CRSI) for mean reversion entries:
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 10 + price > 1d HMA (bullish bias)
   - Short: CRSI > 90 + price < 1d HMA (bearish bias)
   - Proven 75% win rate in range markets

3. KAMA for trend following:
   - Adapts to volatility automatically
   - Long: KAMA_fast > KAMA_slow + ADX > 20
   - Short: KAMA_fast < KAMA_slow + ADX > 20

4. 1d HMA for major trend bias:
   - Only long if price > 1d HMA (bullish major trend)
   - Only short if price < 1d HMA (bearish major trend)

5. ATR trailing stoploss: 2.5x ATR(14)

Why this should work:
- Regime detection avoids whipsaws (major issue in 2022-2023 bear/range)
- Connors RSI excels in range markets (most of 2022-2024)
- KAMA trend following captures 2021 bull and 2025 rallies
- 1d filter ensures alignment with major trend
- 4h timeframe targets 20-50 trades/year (manageable fee drag)
- Discrete sizing (0.25-0.30) minimizes fee churn

Timeframe: 4h (REQUIRED)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_chop_crsi_kama_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - detects trending vs ranging markets.
    CHOP > 61.8 = ranging (mean reversion works)
    CHOP < 38.2 = trending (trend following works)
    Reference: E.W. Dreiss, "The Choppiness Index"
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        # Highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], 
                     abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j],
                     abs(low[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Choppiness formula
        if hh - ll > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - combines momentum, streak, and rank.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    Long: CRSI < 10 (oversold)
    Short: CRSI > 90 (overbought)
    Reference: Connors & Alvarez, "Connors RSI"
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) on close
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_close = 100 - (100 / (1 + rs))
    rsi_close = rsi_close.fillna(50).values
    
    # Streak RSI(2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1
        else:
            streak[i] = streak[i-1]
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0)
    streak_loss = -streak_s.where(streak_s < 0, 0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.nan)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.fillna(50).values
    
    # PercentRank(100) - where does current return rank vs last 100?
    returns = close_s.pct_change().values
    percent_rank = np.zeros(n)
    
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        current = returns[i]
        if len(window) > 0:
            percent_rank[i] = 100 * np.sum(window < current) / len(window)
        else:
            percent_rank[i] = 50.0
    
    # Combine into CRSI
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_kama(close, fast_period=2, slow_period=30, smoothing_period=10):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts to market volatility: fast in trends, slow in ranges.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close_s - close_s.shift(slow_period)).values
    volatility = np.abs(close_s - close_s.shift(1)).values
    
    vol_sum = pd.Series(volatility).rolling(window=slow_period, min_periods=slow_period).sum().values
    
    er = np.zeros(n)
    mask = vol_sum > 0
    er[mask] = change[mask] / vol_sum[mask]
    er = np.clip(er, 0, 1)
    
    # Calculate smoothing constant (SC)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX) - measures trend strength.
    ADX > 25 = strong trend, ADX < 20 = ranging market.
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed values
    tr_smooth = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

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
    Hull Moving Average (HMA) - smoother and more responsive than EMA.
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    # WMA helper
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D HMA for major trend bias
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    # KAMA for trend following
    kama_4h_fast = calculate_kama(close, fast_period=2, slow_period=10, smoothing_period=5)
    kama_4h_slow = calculate_kama(close, fast_period=5, slow_period=30, smoothing_period=10)
    adx_14 = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    HALF_SIZE = 0.14
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]):
            continue
        
        if np.isnan(crsi[i]):
            continue
        
        if np.isnan(kama_4h_fast[i]) or np.isnan(kama_4h_slow[i]):
            continue
        
        if np.isnan(adx_14[i]):
            continue
        
        # === 1D MAJOR TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === REGIME DETECTION ===
        in_range = chop_14[i] > 61.8  # Ranging market
        in_trend = chop_14[i] < 38.2  # Trending market
        # transition zone: 38.2 <= CHOP <= 61.8
        
        # === 4H KAMA TREND ===
        kama_bullish = kama_4h_fast[i] > kama_4h_slow[i]
        kama_bearish = kama_4h_fast[i] < kama_4h_slow[i]
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 20
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15  # Mean reversion long
        crsi_overbought = crsi[i] > 85  # Mean reversion short
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # REGIME 1: RANGE MARKET (CHOP > 61.8) → Mean Reversion with CRSI
        if in_range:
            # Long: CRSI oversold + daily bullish bias
            if crsi_oversold and daily_bullish:
                new_signal = BASE_SIZE
            
            # Short: CRSI overbought + daily bearish bias
            elif crsi_overbought and daily_bearish:
                new_signal = -BASE_SIZE
        
        # REGIME 2: TREND MARKET (CHOP < 38.2) → Trend Following with KAMA
        elif in_trend:
            # Long: KAMA bullish + ADX strong + daily bullish
            if kama_bullish and adx_strong and daily_bullish:
                new_signal = BASE_SIZE
            
            # Short: KAMA bearish + ADX strong + daily bearish
            elif kama_bearish and adx_strong and daily_bearish:
                new_signal = -BASE_SIZE
        
        # REGIME 3: TRANSITION (38.2 <= CHOP <= 61.8) → Reduce size or flat
        else:
            # Only enter with very strong confluence
            if kama_bullish and daily_bullish and crsi_oversold:
                new_signal = HALF_SIZE
            elif kama_bearish and daily_bearish and crsi_overbought:
                new_signal = -HALF_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 100 bars (~17 days on 4h), allow weaker entry
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            if daily_bullish and crsi_oversold:
                new_signal = HALF_SIZE
            elif daily_bearish and crsi_overbought:
                new_signal = -HALF_SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME CHANGE EXIT ===
        regime_change_exit = False
        if in_position and position_side != 0:
            # Exit long if regime changes from range to strong trend bearish
            if position_side > 0 and in_trend and kama_bearish:
                regime_change_exit = True
            # Exit short if regime changes from range to strong trend bullish
            if position_side < 0 and in_trend and kama_bullish:
                regime_change_exit = True
        
        # === CRSI MEAN REVERSION EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long when CRSI becomes overbought (mean reversion complete)
            if position_side > 0 and crsi_overbought:
                crsi_exit = True
            # Exit short when CRSI becomes oversold (mean reversion complete)
            if position_side < 0 and crsi_oversold:
                crsi_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or regime_change_exit or crsi_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals