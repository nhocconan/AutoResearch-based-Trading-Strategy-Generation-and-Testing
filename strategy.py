#!/usr/bin/env python3
"""
Experiment #044: 30m Adaptive Regime Strategy with 4h HMA Trend Bias
Hypothesis: 30m timeframe captures intraday swings while 4h HMA provides regime filter.
Key insight: Most failed 30m strategies used单一 logic (only trend or only mean-revert).
This strategy ADAPTS to market regime using Bollinger Band Width percentile:
- Low BBW (<30th pct) = squeeze/range → mean reversion entries at BB bounds
- High BBW (>70th pct) = trending → momentum breakouts with trend
- 4h HMA bias determines long/short preference (asymmetric positioning)
Position sizing: 0.20-0.30 discrete levels, ATR stoploss at 2.5*ATR
Why this might work: Adapts to changing volatility regimes, avoids whipsaw in ranges
Must generate 10+ trades on train, 3+ on test - entry conditions loosened vs failed experiments.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_adaptive_regime_4h_hma_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / (sma + 1e-10)
    return upper, lower, sma, bandwidth

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / (std + 1e-10)
    return zscore

def calculate_percentile_rank(series, window=100):
    """Calculate rolling percentile rank."""
    def pr(x):
        if len(x) < 2:
            return 0.5
        return (x < x[-1]).sum() / (len(x) - 1)
    pr_values = pd.Series(series).rolling(window=window, min_periods=max(20, window//2)).apply(pr, raw=False).values
    return pr_values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    ema_200 = calculate_ema(close, 200)
    zscore = calculate_zscore(close, 20)
    
    # Bollinger Bands for regime detection
    bb_upper, bb_lower, bb_mid, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    
    # BBW percentile rank for regime detection
    bbw_pct_rank = calculate_percentile_rank(bb_bandwidth, 100)
    
    # HMA on 30m for faster trend
    hma_30m = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.125
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(ema_21[i]) or np.isnan(bb_bandwidth[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - main regime filter
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # 30m trend confirmation
        bull_trend_30m = close[i] > ema_50[i] and ema_21[i] > ema_50[i]
        bear_trend_30m = close[i] < ema_50[i] and ema_21[i] < ema_50[i]
        
        # Long-term trend filter
        above_200 = not np.isnan(ema_200[i]) and close[i] > ema_200[i]
        below_200 = not np.isnan(ema_200[i]) and close[i] < ema_200[i]
        
        # Regime detection via BBW percentile
        low_vol_regime = not np.isnan(bbw_pct_rank[i]) and bbw_pct_rank[i] < 0.30
        high_vol_regime = not np.isnan(bbw_pct_rank[i]) and bbw_pct_rank[i] > 0.70
        normal_regime = not low_vol_regime and not high_vol_regime
        
        # Price position relative to BB
        near_bb_lower = close[i] <= bb_lower[i] * 1.005
        near_bb_upper = close[i] >= bb_upper[i] * 0.995
        near_bb_mid = abs(close[i] - bb_mid[i]) < (bb_upper[i] - bb_lower[i]) * 0.25
        
        # RSI conditions - LOOSENED for more trades
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral_long = 35 < rsi[i] < 55
        rsi_neutral_short = 45 < rsi[i] < 65
        
        # Z-score filter - avoid extreme entries
        zscore_neutral = abs(zscore[i]) < 2.0
        
        # HMA crossover on 30m
        hma_cross_long = False
        hma_cross_short = False
        if i >= 1 and not np.isnan(hma_30m[i]) and not np.isnan(hma_30m[i-1]):
            hma_cross_long = hma_30m[i] > ema_50[i] and hma_30m[i-1] <= ema_50[i-1]
            hma_cross_short = hma_30m[i] < ema_50[i] and hma_30m[i-1] >= ema_50[i-1]
        
        # Price pullback to EMA21
        price_near_ema21_long = close[i] <= ema_21[i] * 1.015 and close[i] >= ema_21[i] * 0.985
        price_near_ema21_short = close[i] >= ema_21[i] * 0.985 and close[i] <= ema_21[i] * 1.015
        
        # Price action: higher low for long, lower high for short
        higher_low = False
        lower_high = False
        if i >= 3:
            higher_low = low[i] > low[i-3]
            lower_high = high[i] < high[i-3]
        
        new_signal = 0.0
        
        # === REGIME 1: LOW VOLATILITY (Range/Mean Reversion) ===
        if low_vol_regime:
            # Long: Price at lower BB + oversold RSI + 4h bullish bias preferred
            if near_bb_lower and rsi_oversold:
                if bull_trend_4h or (bull_trend_30m and above_200):
                    new_signal = SIZE_BASE
                elif zscore[i] < -1.5:
                    new_signal = SIZE_HALF
            
            # Short: Price at upper BB + overbought RSI + 4h bearish bias preferred
            elif near_bb_upper and rsi_overbought:
                if bear_trend_4h or (bear_trend_30m and below_200):
                    new_signal = -SIZE_BASE
                elif zscore[i] > 1.5:
                    new_signal = -SIZE_HALF
        
        # === REGIME 2: HIGH VOLATILITY (Trending/Breakout) ===
        elif high_vol_regime:
            # Long: HMA crossover + trend confirmation + 4h bullish
            if hma_cross_long and bull_trend_30m and bull_trend_4h:
                new_signal = SIZE_BASE
            
            # Short: HMA crossover + trend confirmation + 4h bearish
            elif hma_cross_short and bear_trend_30m and bear_trend_4h:
                new_signal = -SIZE_BASE
            
            # Momentum: Higher low with strong trend
            elif higher_low and bull_trend_4h and rsi[i] > 50:
                new_signal = SIZE_HALF
            
            # Momentum: Lower high with strong trend
            elif lower_high and bear_trend_4h and rsi[i] < 50:
                new_signal = -SIZE_HALF
        
        # === REGIME 3: NORMAL VOLATILITY (Pullback Entries) ===
        else:
            # Long: Pullback to EMA21 in uptrend
            if price_near_ema21_long and rsi_neutral_long:
                if bull_trend_4h and above_200:
                    new_signal = SIZE_BASE
                elif bull_trend_30m:
                    new_signal = SIZE_HALF
            
            # Short: Bounce to EMA21 in downtrend
            elif price_near_ema21_short and rsi_neutral_short:
                if bear_trend_4h and below_200:
                    new_signal = -SIZE_BASE
                elif bear_trend_30m:
                    new_signal = -SIZE_HALF
            
            # Z-score mean reversion
            elif zscore[i] < -1.8 and bull_trend_4h:
                new_signal = SIZE_HALF
            elif zscore[i] > 1.8 and bear_trend_4h:
                new_signal = -SIZE_HALF
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
        if position_side > 0 and entry_price > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        # Short position stoploss
        if position_side < 0 and entry_price > 0:
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals