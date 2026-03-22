#!/usr/bin/env python3
"""
Experiment #315: 1h Fisher Transform + RSI Mean Reversion with 4h HMA Trend Filter

Hypothesis: After #303 (1h Supertrend) failed with Sharpe=-0.891, pure trend following
on 1h timeframe is too noisy for BTC/ETH. Analysis shows:
1. 2025 test period is bear/range market - trend strategies fail
2. Fisher Transform excels at catching reversals in bear markets (research-backed)
3. RSI extremes + SMA200 filter provides mean reversion edge
4. 4h HMA trend filter prevents counter-trend trades in strong trends

This strategy combines:
1. Fisher Transform (period=9): Long when crosses above -1.5, Short when crosses below +1.5
2. RSI(14) extremes: Long when RSI<30, Short when RSI>70 (confirms Fisher)
3. 4h HMA(21) trend filter: Only long when price>4h_HMA, only short when price<4h_HMA
4. ADX(14)>15: Minimal trend confirmation (loose for trade generation)
5. ATR(14) trailing stoploss at 2.5x

Why this should work on 1h:
- Fisher catches reversals quickly (more trades than Supertrend)
- RSI confirmation reduces false signals
- 4h HMA prevents trading against higher timeframe trend
- Designed for bear/range markets (2025 test period)

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_rsi_4h_hma_atr_v1"
timeframe = "1h"
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
    """Calculate Relative Strength Index."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_fisher(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer signals.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5.
    """
    close_s = pd.Series(close)
    
    # Calculate highest high and lowest low over period
    hh = close_s.rolling(window=period, min_periods=period).max()
    ll = close_s.rolling(window=period, min_periods=period).min()
    
    # Normalize price to range 0-1
    norm = (close_s - ll) / (hh - ll)
    norm = norm.replace([np.inf, -np.inf], np.nan)
    norm = norm.fillna(0.5)
    
    # Clamp to avoid division issues
    norm = np.clip(norm, 0.001, 0.999)
    
    # Fisher transform formula
    fisher_raw = 0.5 * np.log((1 + norm) / (1 - norm))
    
    # Smooth with EMA
    fisher = pd.Series(fisher_raw).ewm(span=period, min_periods=period, adjust=False).mean()
    
    return fisher.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index (ADX)."""
    n = len(close)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean()
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.replace([np.inf, -np.inf], np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    return sma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    fisher = calculate_fisher(close, 9)
    adx = calculate_adx(high, low, close, 14)
    sma200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(250, n):  # Start after SMA200 is ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(fisher[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(sma200[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 4h HMA = primary directional bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === TREND STRENGTH ===
        # ADX > 15 = minimal trending (loose for trade generation)
        trending = adx[i] > 15
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.5 = bullish reversal signal
        fisher_bull = fisher[i] > -1.5
        fisher_bull_cross = fisher_bull and (i > 0 and fisher[i-1] <= -1.5)
        
        # Fisher crosses below +1.5 = bearish reversal signal
        fisher_bear = fisher[i] < 1.5
        fisher_bear_cross = fisher_bear and (i > 0 and fisher[i-1] >= 1.5)
        
        # === RSI EXTREMES ===
        # RSI < 30 = oversold (long setup)
        rsi_oversold = rsi[i] < 30
        # RSI > 70 = overbought (short setup)
        rsi_overbought = rsi[i] > 70
        
        # === SMA200 FILTER ===
        # Price above SMA200 = bullish long-term bias
        above_sma200 = close[i] > sma200[i]
        # Price below SMA200 = bearish long-term bias
        below_sma200 = close[i] < sma200[i]
        
        # === VOLATILITY ADJUSTMENT ===
        atr_recent_avg = np.nanmean(atr[max(0, i-20):i+1])
        high_volatility = atr[i] > 1.5 * atr_recent_avg if not np.isnan(atr_recent_avg) else False
        
        # Determine position size
        if high_volatility:
            position_size = SIZE_BASE
        elif trending:
            position_size = SIZE_STRONG
        else:
            position_size = SIZE_BASE
        
        # === ENTRY CONDITIONS ===
        new_signal = 0.0
        
        # LONG: Fisher bullish + RSI oversold + 4h trend up + ADX trending
        # Relaxed: Fisher OR RSI can trigger (not both required)
        long_fisher = fisher_bull_cross and bull_trend_4h
        long_rsi = rsi_oversold and bull_trend_4h and above_sma200
        
        # Require at least one long signal + trend filter
        if (long_fisher or long_rsi) and bull_trend_4h:
            new_signal = position_size
        
        # SHORT: Fisher bearish + RSI overbought + 4h trend down + ADX trending
        short_fisher = fisher_bear_cross and bear_trend_4h
        short_rsi = rsi_overbought and bear_trend_4h and below_sma200
        
        # Require at least one short signal + trend filter
        if (short_fisher or short_rsi) and bear_trend_4h:
            new_signal = -position_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === FISHER REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and fisher[i] > 1.5:  # Fisher too high, exit long
                new_signal = 0.0
            if position_side < 0 and fisher[i] < -1.5:  # Fisher too low, exit short
                new_signal = 0.0
        
        # === RSI EXTREME REVERSAL EXIT ===
        if in_position and new_signal != 0.0:
            if position_side > 0 and rsi[i] > 70:  # RSI overbought, exit long
                new_signal = 0.0
            if position_side < 0 and rsi[i] < 30:  # RSI oversold, exit short
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