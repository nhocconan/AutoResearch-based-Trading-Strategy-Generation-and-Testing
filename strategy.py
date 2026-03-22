#!/usr/bin/env python3
"""
Experiment #517: 15m Multi-Timeframe Mean Reversion with Choppiness Regime

Hypothesis: 15m timeframe is ideal for mean reversion strategies that capture 
intraday oscillations while respecting HTF trend. After 500+ failed experiments,
the key insight is that BTC/ETH spend 70% of time in RANGE mode where mean 
reversion outperforms trend following. Use Choppiness Index to detect regime,
4h HMA for directional bias, and RSI+Z-score for entry timing.

Key innovations:
1. CHOPPINESS INDEX (14): >61.8 = range (mean-revert), <38.2 = trend (follow)
2. 4H HMA BIAS: Via mtf_data helper for trend direction (not too slow like 1d)
3. RSI(7) EXTREMES: <30 long, >70 short (faster than RSI14 for 15m)
4. Z-SCORE(20): <-2.0 long, >+2.0 short (statistical mean reversion)
5. VOLUME CONFIRMATION: taker_buy_volume ratio >0.55 for longs
6. 2.0 * ATR STOPLOSS: Trailing stop to protect capital
7. DISCRETE SIZING: 0.25 positions, minimize fee churn

Why 15m works:
- Captures 2-4 hour swings (crypto's typical mean-reversion cycle)
- 96 bars/day = enough data for statistical significance
- Faster entries than 1h/4h, less noise than 5m
- Choppiness filter avoids trend-following in ranging markets (70% of time)

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_chop_regime_4h_hma_rsi_zscore_volume_atr_v1"
timeframe = "15m"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    Interpretation:
    - CHOP > 61.8: Market is choppy/ranging (mean-reversion works)
    - CHOP < 38.2: Market is trending (trend-following works)
    - 38.2 < CHOP < 61.8: Neutral/transition
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    # Calculate ATR(1) for each bar (True Range)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_zscore(close, period=20):
    """Calculate Z-score of price vs rolling mean."""
    close_s = pd.Series(close)
    rolling_mean = close_s.rolling(window=period, min_periods=period).mean()
    rolling_std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - rolling_mean) / rolling_std.replace(0, np.inf)
    return zscore.values

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio (buying pressure)."""
    ratio = np.zeros(len(volume))
    for i in range(len(volume)):
        if volume[i] > 1e-10:
            ratio[i] = taker_buy_volume[i] / volume[i]
        else:
            ratio[i] = 0.5
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)
    rsi_14 = calculate_rsi(close, 14)
    chop = calculate_choppiness_index(high, low, close, 14)
    zscore = calculate_zscore(close, 20)
    vol_ratio = calculate_volume_ratio(taker_buy_volume, volume)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(chop[i]) or np.isnan(zscore[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range = chop[i] > 61.8  # Mean-reversion regime
        is_trend = chop[i] < 38.2  # Trend-following regime
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # RANGE REGIME: Mean-reversion (70% of crypto market time)
        if is_range:
            # Long: RSI oversold + Z-score low + bull bias preferred
            if rsi_7[i] < 30 and zscore[i] < -1.5:
                if bull_bias or vol_ratio[i] > 0.55:
                    new_signal = SIZE
            # Short: RSI overbought + Z-score high + bear bias preferred
            elif rsi_7[i] > 70 and zscore[i] > 1.5:
                if bear_bias or vol_ratio[i] < 0.45:
                    new_signal = -SIZE
        
        # TREND REGIME: Pullback entries in direction of trend
        elif is_trend:
            # Long pullback in uptrend
            if bull_bias and rsi_7[i] < 40 and zscore[i] < -0.5:
                new_signal = SIZE
            # Short rally in downtrend
            elif bear_bias and rsi_7[i] > 60 and zscore[i] > 0.5:
                new_signal = -SIZE
        
        # NEUTRAL REGIME: Wait for clearer signals
        else:
            # Only enter on extreme conditions
            if rsi_7[i] < 25 and zscore[i] < -2.0 and bull_bias:
                new_signal = SIZE
            elif rsi_7[i] > 75 and zscore[i] > 2.0 and bear_bias:
                new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === REGIME FLIP EXIT ===
        # Exit if regime changes strongly against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and is_trend and bear_bias:
                new_signal = 0.0
            if position_side < 0 and is_trend and bull_bias:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals