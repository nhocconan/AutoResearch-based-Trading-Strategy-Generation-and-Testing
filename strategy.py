#!/usr/bin/env python3
"""
Experiment #1594: 1d Primary + 1w HTF — KAMA Adaptive Trend + Choppiness Regime + RSI

Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility better than 
HMA/EMA, reducing whipsaw in choppy markets while capturing trends efficiently. Combined 
with Choppiness Index regime detection and RSI entry timing, this should outperform 
static MA strategies on daily timeframe.

Why 1d should work better than 12h/6h:
- Daily bars filter intraday noise (BTC/ETH have high intraday volatility)
- Fewer trades = less fee drag (target 20-40 trades/year)
- 1w HTF provides ultra-long-term bias (prevents major counter-trend positions)
- KAMA's adaptive smoothing excels on daily data (Kaufman's original design)

Key innovations vs #1584 (12h Fisher):
1. KAMA instead of HMA - adapts efficiency ratio to volatility regime
2. RSI(7) for faster entry signals than Fisher (more trades guaranteed)
3. ATR ratio for vol spike detection (ATR7/ATR30 > 1.8 = vol expansion)
4. Simpler regime logic: CHOP<40 = trend, CHOP>60 = range (clearer thresholds)
5. 1w HMA slope filter (not just price position) for stronger bias

Entry logic (LOOSE to guarantee ≥30 trades/train):
- LONG trend: 1w_HMA_slope>0 + CHOP<40 + KAMA bullish + RSI(7)<60 crossing up
- SHORT trend: 1w_HMA_slope<0 + CHOP<40 + KAMA bearish + RSI(7)>40 crossing down
- LONG range: CHOP>60 + RSI(7)<25 + price<BB_lower (mean reversion)
- SHORT range: CHOP>60 + RSI(7)>75 + price>BB_upper (mean reversion)

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 1d
Size: 0.25-0.30 discrete (max 0.35)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_chop_regime_rsi_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average - adapts smoothing based on market efficiency
    ER (Efficiency Ratio) = |price change| / sum of absolute price changes
    High ER = trending (less smoothing), Low ER = choppy (more smoothing)
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 1e-10:
            er[i] = price_change / volatility
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    sc = np.full(n, np.nan, dtype=np.float64)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
        else:
            kama[i] = kama[i - 1] if not np.isnan(kama[i - 1]) else close[i]
    
    return kama

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def calculate_hma_slope(close, period=21, lookback=5):
    """
    HMA slope - measures trend direction over lookback periods
    Positive slope = bullish, Negative slope = bearish
    """
    n = len(close)
    if n < period + lookback:
        return np.full(n, np.nan)
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    
    # Calculate slope over lookback
    slope = np.full(n, np.nan, dtype=np.float64)
    for i in range(period + lookback - 1, n):
        if not np.isnan(hma[i]) and not np.isnan(hma[i - lookback]):
            slope[i] = (hma[i] - hma[i - lookback]) / hma[i - lookback] * 100
    
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma_slope(df_1w['close'].values, period=21, lookback=3)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    kama_10 = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_30 = calculate_atr(high, low, close, period=30)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    
    # ATR ratio for vol spike detection
    atr_ratio = np.full(n, np.nan, dtype=np.float64)
    mask = (atr_30 > 1e-10) & (~np.isnan(atr_14)) & (~np.isnan(atr_30))
    atr_ratio[mask] = atr_14[mask] / atr_30[mask]
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track RSI crossings
    prev_rsi_7 = np.nan
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(kama_10[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_trend_regime = chop < 40.0
        is_range_regime = chop > 60.0
        
        # === TREND DIRECTION (1w HMA slope bias) ===
        hma_1w_slope = hma_1w_aligned[i]
        weekly_bullish = hma_1w_slope > 0.5 if not np.isnan(hma_1w_slope) else False
        weekly_bearish = hma_1w_slope < -0.5 if not np.isnan(hma_1w_slope) else False
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama_10[i]
        kama_bearish = close[i] < kama_10[i]
        
        # === RSI SIGNALS (LOOSE thresholds for trade generation) ===
        rsi_val = rsi_7[i]
        rsi_prev = prev_rsi_7 if not np.isnan(prev_rsi_7) else rsi_val
        
        rsi_cross_up_40 = rsi_val > 40 and rsi_prev <= 40
        rsi_cross_down_60 = rsi_val < 60 and rsi_prev >= 60
        rsi_oversold = rsi_val < 30
        rsi_overbought = rsi_val > 70
        
        # === VOLATILITY EXPANSION ===
        vol_expansion = atr_ratio[i] > 1.5 if not np.isnan(atr_ratio[i]) else False
        
        # === BOLLINGER BAND TOUCH ===
        bb_touch_lower = close[i] <= bb_lower[i] * 1.01
        bb_touch_upper = close[i] >= bb_upper[i] * 0.99
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # TREND REGIME: KAMA + RSI + weekly bias
        if is_trend_regime:
            # LONG: weekly bullish + KAMA bullish + RSI crossing up from neutral
            if weekly_bullish and kama_bullish and rsi_cross_up_40:
                desired_signal = SIZE_STRONG if vol_expansion else SIZE_BASE
            
            # SHORT: weekly bearish + KAMA bearish + RSI crossing down from neutral
            elif weekly_bearish and kama_bearish and rsi_cross_down_60:
                desired_signal = -SIZE_STRONG if vol_expansion else -SIZE_BASE
            
            # Fallback: KAMA trend + RSI momentum (no weekly confirmation needed)
            elif kama_bullish and rsi_val > 50 and rsi_val < 70:
                desired_signal = SIZE_BASE
            elif kama_bearish and rsi_val < 50 and rsi_val > 30:
                desired_signal = -SIZE_BASE
        
        # RANGE REGIME: Mean reversion at Bollinger bands
        elif is_range_regime:
            # LONG: RSI oversold + price at BB lower
            if rsi_oversold and bb_touch_lower:
                desired_signal = SIZE_BASE
            
            # SHORT: RSI overbought + price at BB upper
            elif rsi_overbought and bb_touch_upper:
                desired_signal = -SIZE_BASE
            
            # Fallback: RSI extremes alone (looser)
            elif rsi_val < 25:
                desired_signal = SIZE_BASE
            elif rsi_val > 75:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: Use KAMA direction + RSI filter
        else:
            # LONG: KAMA bullish + RSI not overbought
            if kama_bullish and rsi_val < 65:
                desired_signal = SIZE_BASE
            
            # SHORT: KAMA bearish + RSI not oversold
            elif kama_bearish and rsi_val > 35:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
        prev_rsi_7 = rsi_val
    
    return signals