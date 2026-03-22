#!/usr/bin/env python3
"""
Experiment #017: 1d Primary + 1w HTF — KAMA Adaptive Trend with Regime Filter

Hypothesis: Daily timeframe reduces noise and fee drag while capturing major moves.
KAMA (Kaufman Adaptive Moving Average) adapts to volatility - fast in trends, slow in chop.
Combined with ADX for trend strength and Choppiness Index for regime detection.

Why this should work on 1d:
1. KAMA automatically adjusts smoothing based on market efficiency ratio
2. 1w HMA provides major trend bias (only trade WITH weekly trend)
3. ADX > 25 confirms genuine trend (not whipsaw)
4. Choppiness Index < 50 = trending regime (avoid mean-reversion in trends)
5. RSI(14) 40-60 zone for entry timing (not extreme - avoids missing trends)
6. ATR(14) 2.5x trailing stoploss for risk management
7. Target 20-40 trades/year = minimal fee drag (~1-2% annually)

Timeframe: 1d (REQUIRED per experiment)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_adx_chop_1w_hma_v1"
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
    Kaufman Adaptive Moving Average (KAMA)
    
    Adapts smoothing based on market efficiency:
    - High efficiency (trending) = fast EMA
    - Low efficiency (choppy) = slow EMA
    
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = [ER * (fast_sc - slow_sc) + slow_sc]^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    close_s = pd.Series(close)
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio
    price_change = np.abs(close_s.diff(er_period).values)
    noise = np.abs(close_s.diff()).rolling(window=er_period, min_periods=er_period).sum().values
    
    er = price_change / np.where(noise > 0, noise, 1e-10)
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0, 1)
    
    # Smoothing Constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = np.square(er * (fast_sc - slow_sc) + slow_sc)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX)
    Measures trend strength (not direction)
    ADX > 25 = strong trend
    ADX < 20 = weak/range
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - close_s.shift(1).values)
    tr3 = np.abs(low - close_s.shift(1).values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = high_s.diff().values
    minus_dm = -low_s.diff().values
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    # Smoothed DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_s / np.where(tr_s > 0, tr_s, 1e-10)
    minus_di = 100 * minus_dm_s / np.where(tr_s > 0, tr_s, 1e-10)
    
    # DX and ADX
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx = 100 * di_diff / np.where(di_sum > 0, di_sum, 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index
    CHOP > 61.8 = range/choppy
    CHOP < 38.2 = trending
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of ATR(1) over period
    atr1_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # ATR(period)
    atr_period = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Highest High - Lowest Low over period
    hh_ll = high_s.rolling(window=period, min_periods=period).max().values - low_s.rolling(window=period, min_periods=period).min().values
    
    # Choppiness Index
    chop = 100 * (atr1_sum / atr_period) / np.where(hh_ll > 0, hh_ll, 1e-10) * np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_10 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    adx_14 = calculate_adx(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(kama_10[i]) or np.isnan(rsi_14[i]) or np.isnan(adx_14[i]) or np.isnan(chop_14[i]):
            continue
        
        # === 1W TREND BIAS (MAJOR) ===
        # Price above 1w HMA = bullish bias (prefer longs)
        # Price below 1w HMA = bearish bias (prefer shorts)
        trend_1w_bullish = close[i] > hma_1w_21_aligned[i]
        trend_1w_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === KAMA TREND ===
        # Price above KAMA = bullish
        # Price below KAMA = bearish
        kama_bullish = close[i] > kama_10[i]
        kama_bearish = close[i] < kama_10[i]
        
        # KAMA slope (trend direction)
        kama_slope_bullish = kama_10[i] > kama_10[i-1] if i > 0 else False
        kama_slope_bearish = kama_10[i] < kama_10[i-1] if i > 0 else False
        
        # === ADX TREND STRENGTH ===
        # ADX > 25 = strong trend (trend-following mode)
        # ADX < 20 = weak trend (avoid trading or mean-revert)
        strong_trend = adx_14[i] > 22  # Slightly lowered threshold for more trades
        
        # === CHOPPINESS REGIME ===
        # CHOP < 50 = trending (trend-following preferred)
        # CHOP > 60 = choppy (avoid or mean-revert)
        is_trending = chop_14[i] < 50
        is_choppy = chop_14[i] > 60
        
        # === RSI ENTRY TIMING ===
        # In uptrend: look for RSI 40-55 pullback entries (not oversold - trends continue)
        # In downtrend: look for RSI 45-60 bounce entries
        rsi_long_entry = 38 < rsi_14[i] < 58  # Pullback in uptrend
        rsi_short_entry = 42 < rsi_14[i] < 62  # Bounce in downtrend
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Require: 1w bullish + KAMA bullish + KAMA rising + (strong trend OR trending regime) + RSI pullback
        if trend_1w_bullish and kama_bullish and kama_slope_bullish:
            if (strong_trend or is_trending) and rsi_long_entry:
                new_signal = current_size
        
        # SHORT ENTRIES
        # Require: 1w bearish + KAMA bearish + KAMA falling + (strong trend OR trending regime) + RSI bounce
        if trend_1w_bearish and kama_bearish and kama_slope_bearish:
            if (strong_trend or is_trending) and rsi_short_entry:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~2 months on 1d), allow weaker entry
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if trend_1w_bullish and kama_bullish and rsi_14[i] < 50:
                new_signal = current_size * 0.7
            elif trend_1w_bearish and kama_bearish and rsi_14[i] > 50:
                new_signal = -current_size * 0.7
        
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
            if position_side > 0 and trend_1w_bearish and kama_bearish:
                trend_reversal = True
            if position_side < 0 and trend_1w_bullish and kama_bullish:
                trend_reversal = True
        
        # === CHOPPY MARKET EXIT ===
        choppy_exit = False
        if in_position and is_choppy and adx_14[i] < 18:
            choppy_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or choppy_exit:
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