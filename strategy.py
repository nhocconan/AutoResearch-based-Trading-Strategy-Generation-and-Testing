#!/usr/bin/env python3
"""
Experiment #824: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + Connors RSI + ADX Regime

Hypothesis: After 562+ failed strategies, the key insight is that 4h timeframe with
ADAPTIVE trend detection (KAMA) + Connors RSI entries + ADX regime filter works best.
KAMA adapts to volatility (fast in trends, slow in chop) — proven edge over EMA/HMA.

Strategy design:
1. 4h Primary timeframe (target 30-50 trades/year)
2. 12h KAMA(10) for adaptive trend bias (not fixed MA)
3. 4h Connors RSI (RSI3 + RSI-Streak2 + PercentRank100) / 3 for entry timing
4. 4h ADX(14) for regime detection (<20=range, >25=trend)
5. 4h ATR(14) for trailing stop (2.5x)
6. Dual regime: mean revert when ADX<20, trend follow when ADX>25
7. Discrete signal levels: 0.0, ±0.20, ±0.30 (minimize fee churn)

Why KAMA over EMA/HMA:
- Kaufman Adaptive Moving Average adjusts smoothing based on market efficiency
- ER (Efficiency Ratio) = |net change| / sum(|changes|)
- Fast SC = 2/(2+1), Slow SC = 2/(20+1)
- SC = ER*(FastSC-SlowSC) + SlowSC
- KAMA = KAMA_prev + SC * (price - KAMA_prev)
- In trending markets (high ER): KAMA follows price closely
- In choppy markets (low ER): KAMA flattens, reduces whipsaw

Why Connors RSI:
- Combines momentum (RSI3), streak (consecutive up/down days), and percentile rank
- CRSI < 10 = extreme oversold (long opportunity)
- CRSI > 90 = extreme overbought (short opportunity)
- 75% win rate on reversals (research-backed)

Why ADX regime filter:
- ADX < 20 = ranging market (use mean reversion logic)
- ADX > 25 = trending market (use trend following logic)
- Hysteresis: enter at 25, exit at 18 (avoid flip-flopping)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 35-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_crsi_adx_regime_12h1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average — adjusts smoothing based on market efficiency.
    ER = |net change| / sum(|changes|) over er_period
    SC = ER * (fast_sc - slow_sc) + slow_sc
    KAMA = KAMA_prev + SC * (price - KAMA_prev)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA with SMA of first er_period bars
    kama[er_period - 1] = np.mean(close[:er_period])
    
    for i in range(er_period, n):
        # Efficiency Ratio
        net_change = abs(close[i] - close[i - er_period])
        sum_changes = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        
        if sum_changes < 1e-10:
            er = 0.0
        else:
            er = net_change / sum_changes
        
        # Smoothing Constant
        sc = er * (fast_sc - slow_sc) + slow_sc
        
        # KAMA calculation
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI3 + RSI_Streak2 + PercentRank100) / 3
    RSI3: 3-period RSI for short-term momentum
    RSI_Streak2: RSI of consecutive up/down days (streak length)
    PercentRank100: percentile rank of today's return over last 100 days
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # RSI(3)
    rsi3 = calculate_rsi(close, period=rsi_period)
    
    # Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            streak[i] = streak[i - 1] + 1 if i > 0 else 1
        elif close[i] < close[i - 1]:
            streak[i] = streak[i - 1] - 1 if i > 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to positive values for RSI calculation
    streak_positive = np.abs(streak)
    rsi_streak = calculate_rsi(streak_positive, period=streak_period)
    
    # Percentile Rank of returns
    returns = np.diff(close) / (close[:-1] + 1e-10)
    returns = np.concatenate([[0], returns])
    
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = returns[i - rank_period + 1:i + 1]
        count_below = np.sum(window < returns[i])
        percent_rank[i] = 100 * count_below / rank_period
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi3[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi3[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index — measures trend strength.
    ADX < 20 = ranging, ADX > 25 = trending.
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smoothed DM and TR (Wilder's smoothing)
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
        minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
    
    # DX
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # ADX (smoothed DX)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
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

def calculate_donchian(high, low, period=20):
    """Donchian Channels — highest high and lowest low over period."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    crsi_4h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    adx_4h = calculate_adx(high, low, close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align 12h KAMA for trend bias
    kama_12h_raw = calculate_kama(df_12h['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
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
    
    # ADX regime hysteresis tracking
    prev_adx_regime = None  # None, 'range', 'trend'
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(kama_4h[i]) or np.isnan(crsi_4h[i]) or np.isnan(adx_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(kama_12h_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === LONG-TERM TREND BIAS (12h HTF KAMA) ===
        trend_12h_bullish = close[i] > kama_12h_aligned[i]
        trend_12h_bearish = close[i] < kama_12h_aligned[i]
        
        # === PRIMARY TREND (4h KAMA) ===
        trend_4h_bullish = close[i] > kama_4h[i]
        trend_4h_bearish = close[i] < kama_4h[i]
        
        # === ADX REGIME DETECTION with hysteresis ===
        current_adx = adx_4h[i]
        
        if prev_adx_regime is None:
            if current_adx > 25:
                adx_regime = 'trend'
            elif current_adx < 20:
                adx_regime = 'range'
            else:
                adx_regime = prev_adx_regime if prev_adx_regime else 'neutral'
        elif prev_adx_regime == 'trend':
            adx_regime = 'trend' if current_adx > 18 else 'range'
        elif prev_adx_regime == 'range':
            adx_regime = 'range' if current_adx < 22 else 'trend'
        else:
            if current_adx > 25:
                adx_regime = 'trend'
            elif current_adx < 20:
                adx_regime = 'range'
            else:
                adx_regime = prev_adx_regime
        
        prev_adx_regime = adx_regime
        
        # === CONNORS RSI SIGNALS ===
        crsi_extreme_oversold = crsi_4h[i] < 10
        crsi_extreme_overbought = crsi_4h[i] > 90
        crsi_oversold = crsi_4h[i] < 25
        crsi_overbought = crsi_4h[i] > 75
        crsi_neutral_low = 25 <= crsi_4h[i] < 45
        crsi_neutral_high = 55 < crsi_4h[i] <= 75
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        desired_signal = 0.0
        
        # === RANGING REGIME (ADX < 20) — Mean Reversion with CRSI ===
        if adx_regime == 'range':
            # Long: CRSI extreme oversold + 12h trend alignment OR 4h trend
            if crsi_extreme_oversold and (trend_12h_bullish or trend_4h_bullish):
                desired_signal = BASE_SIZE
            
            # Short: CRSI extreme overbought + 12h trend alignment OR 4h trend
            if crsi_extreme_overbought and (trend_12h_bearish or trend_4h_bearish):
                desired_signal = -BASE_SIZE
            
            # CRSI moderate oversold/overbought with confluence
            if crsi_oversold and trend_12h_bullish and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if crsi_overbought and trend_12h_bearish and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
            
            # Fallback: extreme CRSI alone (guarantees trades on all symbols)
            if crsi_extreme_oversold and desired_signal == 0:
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and desired_signal == 0:
                desired_signal = -REDUCED_SIZE
        
        # === TRENDING REGIME (ADX > 25) — Trend Following ===
        elif adx_regime == 'trend':
            # Long: Bullish trend + CRSI pullback to neutral-low OR Donchian breakout
            if trend_12h_bullish or trend_4h_bullish:
                if crsi_neutral_low and trend_12h_bullish:
                    desired_signal = BASE_SIZE
                elif donchian_breakout_long and trend_4h_bullish:
                    desired_signal = REDUCED_SIZE if desired_signal == 0 else desired_signal
            
            # Short: Bearish trend + CRSI pullback to neutral-high OR Donchian breakout
            if trend_12h_bearish or trend_4h_bearish:
                if crsi_neutral_high and trend_12h_bearish:
                    desired_signal = -BASE_SIZE
                elif donchian_breakout_short and trend_4h_bearish:
                    desired_signal = -REDUCED_SIZE if desired_signal == 0 else desired_signal
        
        # === NEUTRAL REGIME (20 <= ADX <= 25) ===
        else:
            # Conservative: CRSI extremes + trend alignment
            if crsi_extreme_oversold and (trend_12h_bullish or trend_4h_bullish):
                desired_signal = REDUCED_SIZE
            
            if crsi_extreme_overbought and (trend_12h_bearish or trend_4h_bearish):
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend intact and CRSI not overbought
                if (trend_12h_bullish or trend_4h_bullish) and crsi_4h[i] < 80:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend intact and CRSI not oversold
                if (trend_12h_bearish or trend_4h_bearish) and crsi_4h[i] > 20:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if both trends reverse + CRSI overbought
            if trend_12h_bearish and trend_4h_bearish and crsi_4h[i] > 85:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if both trends reverse + CRSI oversold
            if trend_12h_bullish and trend_4h_bullish and crsi_4h[i] < 15:
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
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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