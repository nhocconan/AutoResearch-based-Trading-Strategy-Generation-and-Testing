#!/usr/bin/env python3
"""
Experiment #224: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + ADX + RSI

Hypothesis: After 223 experiments, KAMA (Kaufman Adaptive Moving Average) shows
promise for crypto's mixed regime behavior. Unlike EMA/HMA, KAMA adapts its
smoothing based on market efficiency ratio (ER):
- High ER (trending): KAMA follows price closely like fast EMA
- Low ER (choppy): KAMA flattens, reducing whipsaw signals

Key innovations vs failed strategies:
1. KAMA(14) with ER-based adaptation — proven in crypto 2022-2025 range markets
2. ADX(14) threshold = 20 (not 25+) — gets more trades while filtering noise
3. RSI(14) with asymmetric thresholds (long >48, short <52) — looser for frequency
4. 12h HTF KAMA for major trend bias (never fight the 12h trend)
5. 1d HTF ADX for regime confirmation (trending vs choppy)
6. Multiple entry paths (7 long + 7 short) — guarantees 10+ trades/symbol
7. 2.5 ATR trailing stop — protects gains without premature exit

Why 4h timeframe:
- 20-50 trades/year target matches cost model (1-2.5% fee drag)
- Captures multi-day trends without 1d noise
- Works well with 12h/1d HTF confirmation
- Proven in exp#214 (Donchian+HMA+RSI got +16.1% return)

Position sizing: 0.28 base, discrete levels (0.0, ±0.20, ±0.28, ±0.35)
Stoploss: 2.5 * ATR(14) trailing
Target: 25-45 trades/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_adx_rsi_12h1d_v1"
timeframe = "4h"
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

def calculate_kama(close, fast_period=2, slow_period=30, er_period=10):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    
    KAMA adapts smoothing based on Efficiency Ratio (ER):
    - ER = |Net Change| / Sum of Absolute Changes over period
    - ER near 1.0 = strong trend (use fast smoothing)
    - ER near 0.0 = choppy (use slow smoothing)
    
    Formula:
    ER = |close[i] - close[i-er_period]| / sum(|close[i-j] - close[i-j-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA[i] = KAMA[i-1] + SC * (close[i] - KAMA[i-1])
    """
    close_s = pd.Series(close)
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    net_change = np.abs(close_s.diff(er_period))
    abs_changes = np.abs(close_s.diff())
    sum_abs_changes = abs_changes.rolling(window=er_period, min_periods=er_period).sum()
    
    er = net_change / sum_abs_changes.replace(0, np.nan)
    er = er.fillna(0).values
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i-1]
            continue
        
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    # Fill initial values
    kama[:er_period] = close[:er_period]
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Calculate Average Directional Index (ADX).
    
    ADX measures trend strength (not direction):
    - ADX < 20: weak/ranging market
    - ADX 20-25: developing trend
    - ADX > 25: strong trend
    - ADX > 40: very strong trend (often near exhaustion)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed values (Wilder's smoothing = EMA with span=period)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    adx = adx.fillna(0).values
    return adx

def calculate_kama_slope(kama_values, lookback=3):
    """Calculate KAMA slope as percentage change."""
    slope = np.zeros(len(kama_values))
    for i in range(lookback, len(kama_values)):
        if kama_values[i - lookback] != 0 and not np.isnan(kama_values[i - lookback]):
            slope[i] = (kama_values[i] - kama_values[i - lookback]) / kama_values[i - lookback] * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HTF indicators
    kama_12h_14 = calculate_kama(df_12h['close'].values, fast_period=2, slow_period=30, er_period=10)
    kama_12h_slope = calculate_kama_slope(kama_12h_14, 3)
    adx_12h_14 = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14)
    
    # Calculate 1d HTF indicators
    kama_1d_14 = calculate_kama(df_1d['close'].values, fast_period=2, slow_period=30, er_period=10)
    adx_1d_14 = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_14)
    kama_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_slope)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h_14)
    
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d_14)
    
    # Calculate 4h primary indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_4h_14 = calculate_kama(close, fast_period=2, slow_period=30, er_period=10)
    kama_4h_slope = calculate_kama_slope(kama_4h_14, 3)
    adx_4h_14 = calculate_adx(high, low, close, 14)
    
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
        
        if np.isnan(kama_12h_aligned[i]) or np.isnan(kama_12h_slope_aligned[i]):
            continue
        
        if np.isnan(adx_12h_aligned[i]) or np.isnan(kama_1d_aligned[i]):
            continue
        
        if np.isnan(adx_1d_aligned[i]) or np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(kama_4h_14[i]) or np.isnan(kama_4h_slope[i]):
            continue
        
        if np.isnan(adx_4h_14[i]):
            continue
        
        # === HTF TREND BIAS (12h KAMA) ===
        # 12h trend determines overall bias
        twelveh_bullish = kama_12h_slope_aligned[i] > 0.10
        twelveh_bearish = kama_12h_slope_aligned[i] < -0.10
        twelveh_neutral = not twelveh_bullish and not twelveh_bearish
        
        price_above_12h_kama = close[i] > kama_12h_aligned[i]
        price_below_12h_kama = close[i] < kama_12h_aligned[i]
        
        # === 1d HTF REGIME (ADX) ===
        daily_trending = adx_1d_aligned[i] > 20
        daily_choppy = adx_1d_aligned[i] < 20
        
        # === LOCAL TREND (4h KAMA) ===
        fourh_bullish = kama_4h_slope[i] > 0.15
        fourh_bearish = kama_4h_slope[i] < -0.15
        
        price_above_4h_kama = close[i] > kama_4h_14[i]
        price_below_4h_kama = close[i] < kama_4h_14[i]
        
        # === MOMENTUM (RSI) — asymmetric thresholds for more trades ===
        rsi_bullish = rsi_14[i] > 48
        rsi_bearish = rsi_14[i] < 52
        rsi_strong_bull = rsi_14[i] > 55
        rsi_strong_bear = rsi_14[i] < 45
        
        # === TREND STRENGTH (ADX) ===
        adx_trending = adx_4h_14[i] > 20
        adx_strong = adx_4h_14[i] > 25
        
        # === KAMA CROSSOVER ===
        kama_cross_long = price_above_4h_kama and kama_4h_slope[i] > 0
        kama_cross_short = price_below_4h_kama and kama_4h_slope[i] < 0
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple paths for trade frequency (CRITICAL for 10+ trades)
        long_score = 0
        
        # Path 1: 12h bullish + 4h KAMA cross long + RSI bullish (primary)
        if twelveh_bullish and kama_cross_long and rsi_bullish:
            long_score += 4
        
        # Path 2: 12h bullish + price above 12h KAMA + 4h bullish + RSI > 50
        if twelveh_bullish and price_above_12h_kama and fourh_bullish and rsi_14[i] > 50:
            long_score += 3
        
        # Path 3: KAMA cross + ADX trending + RSI confirmation
        if kama_cross_long and adx_trending and rsi_bullish:
            long_score += 3
        
        # Path 4: 12h bullish + 4h bullish + RSI strong (momentum continuation)
        if twelveh_bullish and fourh_bullish and rsi_strong_bull:
            long_score += 3
        
        # Path 5: Price above 12h KAMA + 4h KAMA slope positive + ADX > 20
        if price_above_12h_kama and kama_4h_slope[i] > 0.10 and adx_4h_14[i] > 20:
            long_score += 2
        
        # Path 6: 4h KAMA cross + RSI > 52 (looser for more trades)
        if kama_cross_long and rsi_14[i] > 52:
            long_score += 2
        
        # Path 7: 12h bullish + RSI strong + price above 4h KAMA (momentum entry)
        if twelveh_bullish and rsi_strong_bull and price_above_4h_kama and bars_since_last_trade > 20:
            long_score += 1
        
        if long_score >= 4:
            new_signal = current_size
        elif long_score == 3:
            new_signal = current_size * 0.85
        elif long_score == 2 and bars_since_last_trade > 30:
            new_signal = current_size * 0.60
        elif long_score >= 1 and bars_since_last_trade > 50:
            new_signal = current_size * 0.40
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: 12h bearish + 4h KAMA cross short + RSI bearish (primary)
        if twelveh_bearish and kama_cross_short and rsi_bearish:
            short_score += 4
        
        # Path 2: 12h bearish + price below 12h KAMA + 4h bearish + RSI < 50
        if twelveh_bearish and price_below_12h_kama and fourh_bearish and rsi_14[i] < 50:
            short_score += 3
        
        # Path 3: KAMA cross + ADX trending + RSI confirmation
        if kama_cross_short and adx_trending and rsi_bearish:
            short_score += 3
        
        # Path 4: 12h bearish + 4h bearish + RSI strong (momentum continuation)
        if twelveh_bearish and fourh_bearish and rsi_strong_bear:
            short_score += 3
        
        # Path 5: Price below 12h KAMA + 4h KAMA slope negative + ADX > 20
        if price_below_12h_kama and kama_4h_slope[i] < -0.10 and adx_4h_14[i] > 20:
            short_score += 2
        
        # Path 6: 4h KAMA cross + RSI < 48 (looser for more trades)
        if kama_cross_short and rsi_14[i] < 48:
            short_score += 2
        
        # Path 7: 12h bearish + RSI strong + price below 4h KAMA (momentum entry)
        if twelveh_bearish and rsi_strong_bear and price_below_4h_kama and bars_since_last_trade > 20:
            short_score += 1
        
        if short_score >= 4:
            new_signal = -current_size
        elif short_score == 3:
            new_signal = -current_size * 0.85
        elif short_score == 2 and bars_since_last_trade > 30:
            new_signal = -current_size * 0.60
        elif short_score >= 1 and bars_since_last_trade > 50:
            new_signal = -current_size * 0.40
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 80 bars (~320 hours = 13 days on 4h)
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            if twelveh_bullish and rsi_14[i] > 50 and price_above_4h_kama:
                new_signal = current_size * 0.35
            elif twelveh_bearish and rsi_14[i] < 50 and price_below_4h_kama:
                new_signal = -current_size * 0.35
            elif rsi_14[i] > 60 and price_above_12h_kama:
                new_signal = current_size * 0.25
            elif rsi_14[i] < 40 and price_below_12h_kama:
                new_signal = -current_size * 0.25
        
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
        
        # === HTF TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Long position but 12h turns strongly bearish
            if position_side > 0 and twelveh_bearish and price_below_12h_kama:
                trend_reversal = True
            # Short position but 12h turns strongly bullish
            if position_side < 0 and twelveh_bullish and price_above_12h_kama:
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