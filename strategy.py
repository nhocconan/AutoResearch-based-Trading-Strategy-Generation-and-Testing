#!/usr/bin/env python3
"""
Hypothesis: 30m primary with 4h HTF trend + KAMA adaptive trend + BBW regime filter.
KAMA adapts to volatility (fast in trends, slow in ranges) reducing whipsaws vs EMA.
Bollinger BandWidth percentile detects squeeze/range regimes - reduce size or stay flat.
4h HMA(21) provides robust trend direction without excessive noise.
ATR(14) stoploss at 2.5*ATR + trailing protects during 2022-style crashes.
SIZE=0.30 discrete with regime-based reduction to 0.15 in high uncertainty.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_rsi_bbw_30m_v1"
timeframe = "30m"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market efficiency"""
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_s.diff(er_period))
    volatility = close_s.diff().abs().rolling(er_period, min_periods=er_period).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    # Smoothing constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    return kama

def calculate_hma(close, period):
    """Hull Moving Average - faster response, smoother than EMA"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, adjust=False).mean().values
    wma2 = close_s.ewm(span=period, adjust=False).mean().values
    raw_hma = 2 * wma1 - wma2
    hma = pd.Series(raw_hma).ewm(span=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_bbw(close, high, low, period=20):
    """Bollinger Band Width - volatility/regime indicator"""
    close_s = pd.Series(close)
    sma = close_s.rolling(period, min_periods=period).mean().values
    std = close_s.rolling(period, min_periods=period).std().values
    upper = sma + 2 * std
    lower = sma - 2 * std
    bbw = (upper - lower) / sma
    return bbw, sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    hma_4h = calculate_hma(close_4h, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # 30m indicators - all computed before loop (Rule 8)
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # KAMA(10,2,30) - adaptive trend
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = np.divide(avg_g, avg_l, out=np.ones_like(avg_g), where=avg_l>0)
    rsi = 100 - 100 / (1 + rs)
    
    # ATR(14)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Bollinger Bands for regime detection
    bbw, bb_mid = calculate_bbw(close, high, low, period=20)
    
    # BBW percentile for regime (rolling 100 bars)
    bbw_percentile = pd.Series(bbw).rolling(100, min_periods=50).apply(
        lambda x: np.searchsorted(np.sort(x), x.iloc[-1]) / len(x), raw=False
    ).values
    
    # EMA(50) for additional trend confirmation
    ema50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    SIZE_NORMAL = 0.30
    SIZE_REDUCED = 0.15
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # HTF trend: 4h HMA direction (Rule 2 - use aligned array)
        htf_bullish = close[i] > hma_4h_aligned[i]
        htf_bearish = close[i] < hma_4h_aligned[i]
        
        # Local trend: KAMA slope and price position
        kama_slope = kama[i] - kama[i-5] if i >= 5 else 0
        local_bullish = close[i] > kama[i] and kama_slope > 0
        local_bearish = close[i] < kama[i] and kama_slope < 0
        
        # Regime detection: BBW percentile
        # Low BBW = squeeze (potential breakout coming)
        # High BBW = extended (potential reversal)
        regime_squeeze = bbw_percentile[i] < 0.30 if not np.isnan(bbw_percentile[i]) else False
        regime_extended = bbw_percentile[i] > 0.70 if not np.isnan(bbw_percentile[i]) else False
        
        # Determine position size based on regime
        current_size = SIZE_REDUCED if regime_extended else SIZE_NORMAL
        
        # RSI entry filters
        rsi_ok_long = rsi[i] < 65 and rsi[i] > 40
        rsi_ok_short = rsi[i] > 35 and rsi[i] < 60
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # Price vs BB for mean reversion signals
        bb_upper = bb_mid[i] + 2 * (bbw[i] * bb_mid[i] / 2)
        bb_lower = bb_mid[i] - 2 * (bbw[i] * bb_mid[i] / 2)
        
        # Stoploss and trailing logic (Rule 6)
        if position_side == 1:
            highest_since_entry = max(highest_since_entry, high[i])
            trail_stop = highest_since_entry - 2.5 * atr[i]
            initial_stop = entry_price - 2.5 * atr[i]
            stop_level = max(trail_stop, initial_stop)
            if close[i] < stop_level:
                signals[i] = 0.0
                position_side = 0
                continue
        
        if position_side == -1:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trail_stop = lowest_since_entry + 2.5 * atr[i]
            initial_stop = entry_price + 2.5 * atr[i]
            stop_level = min(trail_stop, initial_stop)
            if close[i] > stop_level:
                signals[i] = 0.0
                position_side = 0
                continue
        
        # Entry logic - only enter when flat
        if position_side == 0:
            # Long entries
            if htf_bullish:
                # Trend continuation: price above KAMA, RSI not overbought
                if local_bullish and rsi_ok_long:
                    signals[i] = current_size
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                # Mean reversion: RSI oversold in uptrend
                elif rsi_oversold and htf_bullish:
                    signals[i] = SIZE_REDUCED
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
            
            # Short entries
            elif htf_bearish:
                # Trend continuation: price below KAMA, RSI not oversold
                if local_bearish and rsi_ok_short:
                    signals[i] = -current_size
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = low[i]
                # Mean reversion: RSI overbought in downtrend
                elif rsi_overbought and htf_bearish:
                    signals[i] = -SIZE_REDUCED
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = low[i]
        else:
            # Hold position - maintain signal
            signals[i] = signals[i-1]
    
    return signals