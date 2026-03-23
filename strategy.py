#!/usr/bin/env python3
"""
Experiment #1077: 1d Primary + 1w HTF — KAMA Adaptive Trend + ADX + Choppiness Regime

Hypothesis: After analyzing 780+ failed experiments, the winning pattern for 1d timeframe uses:
1. KAMA (Kaufman Adaptive Moving Average) — adapts to market efficiency ratio
   Faster in trends, slower in ranges. Better than EMA/HMA for crypto's regime changes.
   Long: price > KAMA + KAMA sloping up | Short: price < KAMA + KAMA sloping down
2. ADX (14) — trend strength confirmation (NOT direction)
   ADX > 20 = trending (follow KAMA direction)
   ADX < 20 = ranging (mean revert at Bollinger bounds)
3. CHOPPINESS INDEX (14) — regime filter for entry timing
   CHOP > 55 = reduce position size (choppy)
   CHOP < 45 = full position size (clean trend)
4. RSI (14) — entry timing within trend
   Long: RSI 35-55 pullback in uptrend | Short: RSI 45-65 rally in downtrend
5. 1w HMA21 macro bias — only trade in direction of weekly trend
6. ATR 3x stoploss — wide stops for daily timeframe

Why this should beat Sharpe=0.612:
- KAMA adapts to crypto's changing volatility (proven in literature)
- ADX threshold of 20 (not 40) ensures enough trades on 1d
- RSI pullback entries (not extremes) generate more signals
- 1d timeframe = 20-40 trades/year (optimal for fee/trade balance)
- Different from all failed CRSI/Fisher/STC strategies

Timeframe: 1d (primary)
HTF: 1w (weekly) — loaded ONCE before loop using mtf_data helper
Position Size: 0.25-0.30 discrete levels (reduced to 0.15 in choppy)
Stoploss: 3.0x ATR trailing (wider for daily)

CRITICAL: Entry conditions MUST be loose enough to generate 10+ trades in train period.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_adx_chop_rsi_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    
    KAMA adapts to market noise via Efficiency Ratio (ER).
    ER = |price change| / sum of absolute price changes
    SC = (ER * (fast SC - slow SC) + slow SC)^2
    KAMA = prior KAMA + SC * (price - prior KAMA)
    
    Benefits:
    - Fast response in trending markets (low noise)
    - Slow response in choppy markets (high noise)
    - Proven in crypto for regime adaptation
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = price_change / noise
        else:
            er[i] = 0.0
    
    # Calculate Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.full(n, np.nan)
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[period] = close[period]
    for i in range(period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1]
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    
    Measures trend strength (not direction).
    ADX > 25 = strong trend
    ADX < 20 = weak trend / ranging
    
    Formula:
    +DM = high - prev_high (if > prev_low - low and > 0)
    -DM = prev_low - low (if > high - prev_high and > 0)
    TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    +DI = 100 * EMA(+DM, period) / EMA(TR, period)
    -DI = 100 * EMA(-DM, period) / EMA(TR, period)
    DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    ADX = EMA(DX, period)
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        # True Range
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        if high[i] - high[i-1] > low[i-1] - low[i] and high[i] - high[i-1] > 0:
            plus_dm[i] = high[i] - high[i-1]
        if low[i-1] - low[i] > high[i] - high[i-1] and low[i-1] - low[i] > 0:
            minus_dm[i] = low[i-1] - low[i]
    
    # Smooth with EMA
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / atr[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / atr[i]
    
    # DX and ADX
    dx = np.full(n, np.nan)
    for i in range(period, n):
        if plus_di[i] + minus_di[i] > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = choppy/range
    CHOP < 38.2 = trending
    38.2 - 61.8 = transition
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    for i in range(period, n):
        if np.isnan(atr_sum[i]) or np.isnan(hh[i]) or np.isnan(ll[i]):
            continue
        price_range = hh[i] - ll[i]
        if price_range > 1e-10 and atr_sum[i] > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum[i] / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(series, period):
    """Hull Moving Average"""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands for mean reversion entries"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA21 for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    kama = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(kama[i]) or np.isnan(adx[i]):
            continue
        if np.isnan(chop[i]) or np.isnan(rsi[i]):
            continue
        if np.isnan(atr[i]) or np.isnan(hma_1w_aligned[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === VOLATILITY/CHOP REGIME (Position Sizing) ===
        is_choppy = chop[i] > 55.0
        current_size = REDUCED_SIZE if is_choppy else BASE_SIZE
        
        # === MACRO TREND (1w HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === KAMA TREND DIRECTION ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # KAMA slope (compare to 3 bars ago)
        kama_slope_up = kama[i] > kama[i-3] if i >= 3 and not np.isnan(kama[i-3]) else False
        kama_slope_down = kama[i] < kama[i-3] if i >= 3 and not np.isnan(kama[i-3]) else False
        
        # === ADX TREND STRENGTH ===
        is_trending = adx[i] > 20.0
        is_ranging = adx[i] <= 20.0
        
        # === RSI ENTRY TIMING ===
        rsi_pullback_long = 35.0 <= rsi[i] <= 55.0
        rsi_rally_short = 45.0 <= rsi[i] <= 65.0
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        
        desired_signal = 0.0
        
        # === TRENDING REGIME (ADX > 20): Follow KAMA direction ===
        if is_trending:
            # Long: KAMA bullish + RSI pullback + macro bullish
            if kama_bull and kama_slope_up and rsi_pullback_long and macro_bull:
                desired_signal = current_size
            # Long: KAMA bullish + RSI oversold + macro bullish (stronger signal)
            elif kama_bull and kama_slope_up and rsi_oversold and macro_bull:
                desired_signal = current_size
            
            # Short: KAMA bearish + RSI rally + macro bearish
            elif kama_bear and kama_slope_down and rsi_rally_short and macro_bear:
                desired_signal = -current_size
            # Short: KAMA bearish + RSI overbought + macro bearish (stronger signal)
            elif kama_bear and kama_slope_down and rsi_overbought and macro_bear:
                desired_signal = -current_size
        
        # === RANGING REGIME (ADX <= 20): Mean revert at Bollinger bounds ===
        else:
            # Long at lower BB + RSI oversold + macro not strongly bearish
            if close[i] <= bb_lower[i] * 1.002 and rsi_oversold and not macro_bear:
                desired_signal = REDUCED_SIZE
            # Short at upper BB + RSI overbought + macro not strongly bullish
            elif close[i] >= bb_upper[i] * 0.998 and rsi_overbought and not macro_bull:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 3.0x for daily) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if setup intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if KAMA still bullish or RSI not overbought
                if kama_bull and rsi[i] < 70.0:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if KAMA still bearish or RSI not oversold
                if kama_bear and rsi[i] > 30.0:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if KAMA reverses bearish AND RSI overbought
            if kama_bear and rsi_overbought:
                desired_signal = 0.0
            # Exit long if macro reverses strongly bearish
            if macro_bear and adx[i] > 25.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if KAMA reverses bullish AND RSI oversold
            if kama_bull and rsi_oversold:
                desired_signal = 0.0
            # Exit short if macro reverses strongly bullish
            if macro_bull and adx[i] > 25.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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