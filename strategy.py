#!/usr/bin/env python3
"""
Experiment #212: 12h Primary + 1d/1w HTF — KAMA Adaptive Trend + ADX Regime Switch

Hypothesis: Previous Connors RSI + BB strategies failed because they're too mean-reversion
focused in trending markets. KAMA (Kaufman Adaptive Moving Average) adapts to volatility
automatically - fast in trends, slow in chop. Combined with ADX for trend strength and
Choppiness for regime detection, this should capture trends while avoiding whipsaws.

Key innovations:
1. KAMA(10,2,30) - adapts speed based on Efficiency Ratio (ER)
2. ADX(14) > 25 confirms genuine trend (not noise)
3. Choppiness Index regime: >55 = range (mean revert), <45 = trend (follow)
4. 1d KAMA slope for major trend bias
5. 1w HMA for secular trend filter (avoid counter-secular trades)
6. Dual entry paths: trend follow OR mean revert based on regime

Why this should work:
- KAMA outperforms EMA/HMA in mixed regime markets (academic literature)
- ADX filter prevents entries during low-volatility chop
- 12h timeframe = 25-50 trades/year target (low fee drag)
- Regime-switching logic adapts to market conditions
- Multiple entry paths ensure sufficient trade count

Timeframe: 12h (REQUIRED)
HTF: 1d + 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.28 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-50/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_adx_chop_1d1w_v1"
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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise via Efficiency Ratio (ER).
    ER = |net change| / sum of absolute changes over period
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = prior_KAMA + SC * (price - prior_KAMA)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Calculate Efficiency Ratio (ER)
    net_change = np.abs(close_s.diff(er_period).values)
    sum_changes = pd.Series(np.abs(close_s.diff())).rolling(window=er_period, min_periods=er_period).sum().values
    
    er = np.zeros(n)
    mask = sum_changes > 0
    er[mask] = net_change / sum_changes[mask]
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    ADX measures trend strength (not direction).
    ADX > 25 = trending, ADX < 20 = ranging
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
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Smoothed values
    tr_smooth = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / tr_smooth)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / tr_smooth)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(0).values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_values = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_kama_slope(kama_values, lookback=5):
    """Calculate KAMA slope as percentage change."""
    slope = np.zeros(len(kama_values))
    for i in range(lookback, len(kama_values)):
        if kama_values[i - lookback] != 0:
            slope[i] = (kama_values[i] - kama_values[i - lookback]) / kama_values[i - lookback] * 100
    return slope

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF indicators
    kama_1d_10 = calculate_kama(df_1d['close'].values, 10, 2, 30)
    kama_1d_slope = calculate_kama_slope(kama_1d_10, 5)
    
    # Calculate 1w HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_1d_10_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_10)
    kama_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_slope)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_12h_10 = calculate_kama(close, 10, 2, 30)
    adx_14 = calculate_adx(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # KAMA crossover signals
    kama_cross_up = np.zeros(n)
    kama_cross_down = np.zeros(n)
    for i in range(2, n):
        if kama_12h_10[i-1] < close[i-1] and kama_12h_10[i] > close[i]:
            kama_cross_down[i] = 1
        if kama_12h_10[i-1] > close[i-1] and kama_12h_10[i] < close[i]:
            kama_cross_up[i] = 1
    
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
        
        if np.isnan(kama_1d_10_aligned[i]) or np.isnan(kama_1d_slope_aligned[i]):
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(kama_12h_10[i]) or np.isnan(donchian_upper[i]):
            continue
        
        # === 1W SECULAR TREND (major bias) ===
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === 1D TREND BIAS ===
        trend_1d_bullish = kama_1d_slope_aligned[i] > 0.5
        trend_1d_bearish = kama_1d_slope_aligned[i] < -0.5
        price_above_1d_kama = close[i] > kama_1d_10_aligned[i]
        price_below_1d_kama = close[i] < kama_1d_10_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === ADX TREND STRENGTH ===
        adx_strong = adx_14[i] > 25
        adx_weak = adx_14[i] < 20
        
        # === KAMA POSITION ===
        price_above_kama = close[i] > kama_12h_10[i]
        price_below_kama = close[i] < kama_12h_10[i]
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        rsi_extreme_low = rsi_14[i] < 25
        rsi_extreme_high = rsi_14[i] > 75
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_upper[i] * 0.995
        donchian_breakout_down = close[i] < donchian_lower[i] * 1.005
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        if is_range_market:
            current_size = BASE_SIZE * 0.8  # Smaller in range (more whipsaw risk)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple confluence paths
        long_score = 0
        long_confidence = 0
        
        # Path 1: Trend market + ADX strong + price above KAMA + 1d bullish
        if is_trend_market and adx_strong and price_above_kama and trend_1d_bullish:
            long_score += 3
            long_confidence += 2
        
        # Path 2: Range market + RSI oversold + price below KAMA (mean revert)
        if is_range_market and rsi_oversold and price_below_kama:
            long_score += 3
            long_confidence += 1
        
        # Path 3: Donchian breakout up + ADX strong + 1w bullish
        if donchian_breakout_up and adx_strong and price_above_1w_hma:
            long_score += 3
            long_confidence += 2
        
        # Path 4: KAMA cross up + RSI not overbought + 1d bullish
        if kama_cross_up[i] > 0 and rsi_14[i] < 60 and trend_1d_bullish:
            long_score += 2
            long_confidence += 1
        
        # Path 5: Price above 1d KAMA + RSI pullback (dip buy in uptrend)
        if price_above_1d_kama and rsi_14[i] < 45 and trend_1d_bullish:
            long_score += 2
            long_confidence += 1
        
        # Path 6: Simple KAMA cross with 1w support
        if kama_cross_up[i] > 0 and price_above_1w_hma:
            long_score += 1
            long_confidence += 1
        
        if long_score >= 3:
            new_signal = current_size
        elif long_score >= 2 and long_confidence >= 2:
            new_signal = current_size
        elif long_score >= 2 and bars_since_last_trade > 60:
            new_signal = current_size * 0.6
        
        # SHORT ENTRIES
        short_score = 0
        short_confidence = 0
        
        # Path 1: Trend market + ADX strong + price below KAMA + 1d bearish
        if is_trend_market and adx_strong and price_below_kama and trend_1d_bearish:
            short_score += 3
            short_confidence += 2
        
        # Path 2: Range market + RSI overbought + price above KAMA (mean revert)
        if is_range_market and rsi_overbought and price_above_kama:
            short_score += 3
            short_confidence += 1
        
        # Path 3: Donchian breakout down + ADX strong + 1w bearish
        if donchian_breakout_down and adx_strong and price_below_1w_hma:
            short_score += 3
            short_confidence += 2
        
        # Path 4: KAMA cross down + RSI not oversold + 1d bearish
        if kama_cross_down[i] > 0 and rsi_14[i] > 40 and trend_1d_bearish:
            short_score += 2
            short_confidence += 1
        
        # Path 5: Price below 1d KAMA + RSI rally (sell rip in downtrend)
        if price_below_1d_kama and rsi_14[i] > 55 and trend_1d_bearish:
            short_score += 2
            short_confidence += 1
        
        # Path 6: Simple KAMA cross with 1w resistance
        if kama_cross_down[i] > 0 and price_below_1w_hma:
            short_score += 1
            short_confidence += 1
        
        if short_score >= 3:
            new_signal = -current_size
        elif short_score >= 2 and short_confidence >= 2:
            new_signal = -current_size
        elif short_score >= 2 and bars_since_last_trade > 60:
            new_signal = -current_size * 0.6
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~60 days on 12h)
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and rsi_14[i] < 45:
                new_signal = current_size * 0.4
            elif trend_1d_bearish and rsi_14[i] > 55:
                new_signal = -current_size * 0.4
            elif price_above_1w_hma and rsi_14[i] < 40:
                new_signal = current_size * 0.35
            elif price_below_1w_hma and rsi_14[i] > 60:
                new_signal = -current_size * 0.35
        
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
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if regime shifts to strong bearish trend
            if position_side > 0 and is_trend_market and trend_1d_bearish and adx_strong:
                regime_reversal = True
            # Exit short if regime shifts to strong bullish trend
            if position_side < 0 and is_trend_market and trend_1d_bullish and adx_strong:
                regime_reversal = True
        
        # === RSI EXTREME EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_14[i] > 75:
                rsi_exit = True  # Take profit on long at RSI extreme
            if position_side < 0 and rsi_14[i] < 25:
                rsi_exit = True  # Take profit on short at RSI extreme
        
        if stoploss_triggered or regime_reversal or rsi_exit:
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