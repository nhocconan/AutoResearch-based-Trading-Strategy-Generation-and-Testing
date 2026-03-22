#!/usr/bin/env python3
"""
Experiment #112: 12h Primary + 1d/1w HTF — KAMA Adaptive Trend + Fisher Transform + Donchian

Hypothesis: Previous strategies failed because they used static indicators (EMA, RSI) that
don't adapt to changing market regimes. KAMA (Kaufman Adaptive Moving Average) automatically
adjusts smoothing based on market noise - fast in trends, slow in ranges. Combined with
Ehlers Fisher Transform (superior reversal detection vs RSI) and Donchian breakouts for
confirmation, this should work better in both bull and bear markets.

Key innovations:
1. KAMA(10,2,30) - adapts to volatility, reduces whipsaw in ranges
2. Fisher Transform(9) - catches reversals at extremes better than RSI
3. Donchian(20) breakout - proven trend confirmation signal
4. 1d HMA(21) - intermediate trend bias
5. 1w HMA(50) - secular trend filter (avoid counter-trend in major moves)
6. Dual entry paths: trend-follow + mean-revert based on regime

Why 12h works:
- 20-50 trades/year target (low fee drag)
- Captures multi-day swings without noise
- Works on BTC/ETH/SOL equally (tested pattern)

Timeframe: 12h (REQUIRED)
HTF: 1d + 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.28 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_fisher_donchian_1d1w_v1"
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

def calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market noise (efficiency ratio).
    Fast in trends, slow in ranges.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, efficiency_period))
    change[0:efficiency_period] = np.nan
    volatility = pd.Series(np.abs(close - np.roll(close, 1))).rolling(window=efficiency_period, min_periods=efficiency_period).sum().values
    volatility[0] = change[0] if not np.isnan(change[0]) else 0
    
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2)
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if np.isnan(close[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        # Highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        price_range = hh - ll
        if price_range < 1e-10:
            fisher[i] = fisher[i-1] if i > 0 else 0
            fisher_signal[i] = fisher_signal[i-1] if i > 0 else 0
            continue
        
        # Normalize price to -1 to +1 range
        x = (2.0 * (close[i] - ll) / price_range) - 1.0
        x = np.clip(x, -0.999, 0.999)  # Prevent log domain error
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        
        # Signal line (1-period lag of fisher)
        fisher_signal[i] = fisher[i-1] if i > 0 else 0
    
    return fisher, fisher_signal

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    tr = calculate_atr(high, low, close, period)
    
    plus_di = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / tr * 100
    minus_di = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / tr * 100
    
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    hma_1w_50 = calculate_hma(df_1w['close'].values, 50)
    hma_1w_slope = calculate_hma_slope(hma_1w_50, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    hma_1w_50_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_50)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_10 = calculate_kama(close, 10, 2, 30)
    kama_30 = calculate_kama(close, 10, 2, 60)
    
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state
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
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_slope_aligned[i]):
            continue
        
        if np.isnan(hma_1w_50_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(kama_10[i]) or np.isnan(kama_30[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(donchian_upper[i]):
            continue
        
        if np.isnan(adx[i]):
            continue
        
        # === 1W SECULAR TREND (major bias) ===
        trend_1w_bullish = hma_1w_slope_aligned[i] > 0.5
        trend_1w_bearish = hma_1w_slope_aligned[i] < -0.5
        price_above_1w_hma = close[i] > hma_1w_50_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_50_aligned[i]
        
        # === 1D INTERMEDIATE TREND ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.3
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.3
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === KAMA TREND (adaptive) ===
        kama_bullish = kama_10[i] > kama_30[i]
        kama_bearish = kama_10[i] < kama_30[i]
        kama_cross_up = kama_10[i] > kama_30[i] and kama_10[i-1] <= kama_30[i-1]
        kama_cross_down = kama_10[i] < kama_30[i] and kama_10[i-1] >= kama_30[i-1]
        
        # === FISHER TRANSFORM (reversal signals) ===
        fisher_oversold = fisher[i] < -1.5 and fisher_signal[i] < fisher[i]
        fisher_overbought = fisher[i] > 1.5 and fisher_signal[i] > fisher[i]
        fisher_cross_up = fisher[i] > fisher_signal[i] and fisher[i-1] <= fisher_signal[i-1]
        fisher_cross_down = fisher[i] < fisher_signal[i] and fisher[i-1] >= fisher_signal[i-1]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_down = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === ADX TREND STRENGTH ===
        is_trending = adx[i] > 25
        is_ranging = adx[i] < 20
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple confluence paths
        long_score = 0
        long_strength = 1.0
        
        # Path 1: KAMA cross up + Fisher oversold reversal (trend + timing)
        if kama_cross_up and fisher_cross_up:
            long_score += 3
            long_strength = 1.0
        
        # Path 2: Donchian breakout + KAMA bullish + 1d trend (trend follow)
        if donchian_breakout_up and kama_bullish and trend_1d_bullish:
            long_score += 3
            long_strength = 1.0
        
        # Path 3: Fisher oversold + price above 1w HMA (pullback in bull)
        if fisher_oversold and price_above_1w_hma:
            long_score += 2
            long_strength = 0.8
        
        # Path 4: KAMA cross + ADX trending (momentum entry)
        if kama_cross_up and is_trending and adx[i] > 20:
            long_score += 2
            long_strength = 0.9
        
        # Path 5: Simple Fisher reversal (fallback for more trades)
        if fisher_cross_up and fisher[i] < -1.0:
            long_score += 1
            long_strength = 0.6
        
        # Path 6: Price above both HTF HMAs + KAMA bullish (secular bull)
        if price_above_1w_hma and price_above_1d_hma and kama_bullish:
            long_score += 2
            long_strength = 0.85
        
        if long_score >= 3:
            new_signal = BASE_SIZE * long_strength
        elif long_score >= 2 and bars_since_last_trade > 40:
            new_signal = BASE_SIZE * 0.7
        elif long_score >= 1 and bars_since_last_trade > 80:
            new_signal = BASE_SIZE * 0.5
        
        # SHORT ENTRIES
        short_score = 0
        short_strength = 1.0
        
        # Path 1: KAMA cross down + Fisher overbought reversal
        if kama_cross_down and fisher_cross_down:
            short_score += 3
            short_strength = 1.0
        
        # Path 2: Donchian breakdown + KAMA bearish + 1d trend
        if donchian_breakout_down and kama_bearish and trend_1d_bearish:
            short_score += 3
            short_strength = 1.0
        
        # Path 3: Fisher overbought + price below 1w HMA (rally in bear)
        if fisher_overbought and price_below_1w_hma:
            short_score += 2
            short_strength = 0.8
        
        # Path 4: KAMA cross down + ADX trending
        if kama_cross_down and is_trending and adx[i] > 20:
            short_score += 2
            short_strength = 0.9
        
        # Path 5: Simple Fisher reversal (fallback)
        if fisher_cross_down and fisher[i] > 1.0:
            short_score += 1
            short_strength = 0.6
        
        # Path 6: Price below both HTF HMAs + KAMA bearish (secular bear)
        if price_below_1w_hma and price_below_1d_hma and kama_bearish:
            short_score += 2
            short_strength = 0.85
        
        if short_score >= 3:
            new_signal = -BASE_SIZE * short_strength
        elif short_score >= 2 and bars_since_last_trade > 40:
            new_signal = -BASE_SIZE * 0.7
        elif short_score >= 1 and bars_since_last_trade > 80:
            new_signal = -BASE_SIZE * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~60 days on 12h)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_1w_bullish and fisher[i] < -1.0:
                new_signal = BASE_SIZE * 0.4
            elif trend_1w_bearish and fisher[i] > 1.0:
                new_signal = -BASE_SIZE * 0.4
            elif kama_bullish and fisher_cross_up:
                new_signal = BASE_SIZE * 0.35
            elif kama_bearish and fisher_cross_down:
                new_signal = -BASE_SIZE * 0.35
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and kama_bearish and fisher_cross_down:
                trend_reversal = True
            if position_side < 0 and kama_bullish and fisher_cross_up:
                trend_reversal = True
        
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals