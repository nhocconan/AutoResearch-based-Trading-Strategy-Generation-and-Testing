#!/usr/bin/env python3
"""
Experiment #102: 1d Regime-Adaptive + 1w HMA Bias + Choppiness Filter + RSI Mean Reversion
Hypothesis: Daily timeframe benefits from regime detection to switch between trend-following
and mean-reversion based on market conditions. Choppiness Index (CHOP) distinguishes ranging
vs trending markets. In trends (CHOP<38.2), follow KAMA/EMA. In ranges (CHOP>61.8), mean
revert with RSI extremes. 1w HMA provides ultra-stable long-term bias. This should work
better in 2025 bear/range market than pure trend strategies.

Why this might beat Sharpe=0.436:
- Regime adaptation handles both bull (2021) and bear/range (2022, 2025) markets
- CHOP filter prevents trend-following whipsaws in choppy periods
- RSI mean reversion captures reversals in ranges (common in 2025)
- 1w HMA bias prevents counter-trend trades against major trend
- Discrete sizing (0.20/0.30) minimizes fee churn on 1d timeframe
- ATR stoploss (2.5x) protects against major crashes like 2022

Timeframe: 1d (REQUIRED), HTF: 1w via mtf_data helper (call ONCE before loop).
Position sizing: 0.20 base, 0.30 strong signals. Stoploss at 2.5*ATR.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_adaptive_1w_hma_chop_rsi_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA)."""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    er[:er_period] = np.nan
    
    # Calculate Smoothing Constant (SC)
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    sc[:er_period] = np.nan
    
    # Calculate KAMA
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] ** 2 * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend following)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        atr_sum = 0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], 
                     abs(high[j] - close[j-1]), 
                     abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower, sma

def calculate_ema(close, period):
    """Calculate EMA."""
    return pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    chop = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(rsi[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1w HMA = higher timeframe trend bias (ultra-stable)
        bull_trend_1w = close[i] > hma_1w_aligned[i]
        bear_trend_1w = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        # CHOP > 61.8 = ranging market (use mean reversion)
        # CHOP < 38.2 = trending market (use trend following)
        # 38.2 < CHOP < 61.8 = transition zone (reduce position size or stay flat)
        ranging_market = chop[i] > 61.8
        trending_market = chop[i] < 38.2
        transition_zone = 38.2 <= chop[i] <= 61.8
        
        # === KAMA TREND SIGNAL ===
        kama_bullish = close[i] > kama[i] and kama[i] > ema_50[i]
        kama_bearish = close[i] < kama[i] and kama[i] < ema_50[i]
        
        # === EMA ALIGNMENT ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === RSI MEAN REVERSION SIGNALS ===
        # Oversold (long opportunity in range)
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        # RSI momentum
        rsi_momentum_long = rsi[i] > 45 and rsi[i] < 60
        rsi_momentum_short = rsi[i] < 55 and rsi[i] > 40
        
        # === BOLLINGER BAND SIGNALS ===
        bb_long = close[i] < bb_lower[i]  # Price at lower band
        bb_short = close[i] > bb_upper[i]  # Price at upper band
        
        new_signal = 0.0
        
        # === TRENDING REGIME (CHOP < 38.2) - Follow Trend ===
        if trending_market:
            # Long: KAMA bullish + 1w bullish + EMA alignment + RSI momentum
            if kama_bullish and bull_trend_1w and ema_bullish:
                if rsi_momentum_long:
                    new_signal = SIZE_STRONG
                else:
                    new_signal = SIZE_BASE
            
            # Short: KAMA bearish + 1w bearish + EMA alignment + RSI momentum
            elif kama_bearish and bear_trend_1w and ema_bearish:
                if rsi_momentum_short:
                    new_signal = -SIZE_STRONG
                else:
                    new_signal = -SIZE_BASE
            
            # Simpler trend entry (ensures trades on all symbols)
            elif kama_bullish and bull_trend_1w:
                new_signal = SIZE_BASE
            elif kama_bearish and bear_trend_1w:
                new_signal = -SIZE_BASE
        
        # === RANGING REGIME (CHOP > 61.8) - Mean Reversion ===
        elif ranging_market:
            # Long: RSI oversold + price at BB lower + 1w bullish bias
            if rsi_oversold and bb_long:
                if bull_trend_1w:
                    new_signal = SIZE_STRONG
                else:
                    new_signal = SIZE_BASE
            
            # Short: RSI overbought + price at BB upper + 1w bearish bias
            elif rsi_overbought and bb_short:
                if bear_trend_1w:
                    new_signal = -SIZE_STRONG
                else:
                    new_signal = -SIZE_BASE
            
            # Simpler mean reversion (ensures trades)
            elif rsi_oversold and bull_trend_1w:
                new_signal = SIZE_BASE
            elif rsi_overbought and bear_trend_1w:
                new_signal = -SIZE_BASE
        
        # === TRANSITION ZONE (38.2 <= CHOP <= 61.8) - Reduced Signals ===
        elif transition_zone:
            # Only take strong signals with 1w confirmation
            if kama_bullish and bull_trend_1w and rsi_momentum_long:
                new_signal = SIZE_BASE
            elif kama_bearish and bear_trend_1w and rsi_momentum_short:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR for 1d ===
        # Update trailing highs/lows for active positions
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            # Trailing stop: 2.5 * ATR below highest close
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            # Trailing stop: 2.5 * ATR above lowest close
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0  # Stoploss hit
        
        # Update position tracking
        # Entering new position
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Reversing position
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Exiting position
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals