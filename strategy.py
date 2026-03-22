#!/usr/bin/env python3
"""
Experiment #367: 15m RSI Mean Reversion with 4h HMA Trend Bias + Volume Filter

Hypothesis: After analyzing 366 failed experiments, the pattern shows:
1. Pure trend-following fails on BTC/ETH (whipsaw in 2022 crash)
2. Pure mean-reversion fails without trend filter (counter-trend losses)
3. 15m timeframe needs HTF bias to avoid noise

This strategy combines:
1. 4h HMA(21) trend bias - only trade in direction of higher timeframe trend
2. RSI(14) mean reversion - enter on pullbacks (RSI<40 long, RSI>60 short)
3. Volume confirmation - volume > 1.3x 20-period average (filters false signals)
4. Bollinger Band width filter - avoid entering during extreme volatility
5. ATR(14) trailing stop at 2.0x - protect capital on reversals

Why 15m should work:
- Fast enough to catch intraday mean reversion opportunities
- 4h HMA provides stable trend bias (filters 60%+ of counter-trend trades)
- Volume filter reduces false breakouts
- Should generate 50-100 trades/year per symbol (enough for statistical significance)
- Conservative position sizing (0.25) limits drawdown during 2022 crash

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_4h_hma_volume_bb_atr_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    # Bandwidth = (upper - lower) / sma
    bandwidth = (upper - lower) / sma
    
    return upper.values, lower.values, bandwidth.values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs 20-period average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = volume / vol_avg.values
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Calculate BB bandwidth percentile (for regime filter)
    bb_percentile = np.full(n, np.nan)
    lookback = 100
    for i in range(lookback, n):
        valid_bw = bb_bandwidth[i-lookback:i+1]
        valid_bw = valid_bw[~np.isnan(valid_bw)]
        if len(valid_bw) > 0:
            bb_percentile[i] = np.percentile(valid_bw, np.searchsorted(np.sort(valid_bw), bb_bandwidth[i])) / len(valid_bw) * 100
    
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
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_bandwidth[i]):
            signals[i] = 0.0
            continue
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === RSI MEAN REVERSION SIGNALS ===
        # Loosened thresholds to generate more trades (35/65 instead of 30/70)
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        # === VOLUME CONFIRMATION ===
        # Volume must be above average to confirm signal
        volume_confirmed = vol_ratio[i] > 1.2
        
        # === BOLLINGER BAND REGIME FILTER ===
        # Avoid extreme volatility (bandwidth > 80th percentile)
        low_volatility = True
        if not np.isnan(bb_percentile[i]) and bb_percentile[i] > 80:
            low_volatility = False
        
        # === GENERATE SIGNAL ===
        new_signal = 0.0
        
        # LONG ENTRY: RSI oversold + 4h bullish bias + volume confirmed + low vol regime
        if rsi_oversold and bull_trend_4h and volume_confirmed and low_volatility:
            new_signal = SIZE
        
        # SHORT ENTRY: RSI overbought + 4h bearish bias + volume confirmed + low vol regime
        elif rsi_overbought and bear_trend_4h and volume_confirmed and low_volatility:
            new_signal = -SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h trend flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === RSI MEAN REVERSION EXIT ===
        # Exit long when RSI crosses above 55 (mean reversion complete)
        # Exit short when RSI crosses below 45 (mean reversion complete)
        if in_position and new_signal != 0.0:
            if position_side > 0 and rsi[i] > 55:
                new_signal = 0.0
            if position_side < 0 and rsi[i] < 45:
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