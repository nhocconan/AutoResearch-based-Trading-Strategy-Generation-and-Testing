#!/usr/bin/env python3
"""
Experiment #034: 4h Fisher Transform + Daily HMA Regime + ATR Volatility Scaling
Hypothesis: 4h timeframe captures multi-day swings while avoiding intraday noise.
Ehlers Fisher Transform (period=9) excels at detecting reversals in bear/range markets.
Daily HMA provides major trend regime filter (avoid counter-trend trades).
ATR volatility scaling reduces position size during high volatility (2022 crash protection).
Multiple Fisher entry triggers (cross above -1.5 long, cross below +1.5 short) ensure ≥10 trades.
Position sizing 0.25-0.30 with vol_scaling and 2.5x ATR stoploss controls drawdown.
This differs from failed 4h strategies by using Fisher instead of Supertrend/RSI/MACD.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_daily_hma_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to -1 to +1 range.
    Sharp turning points indicate reversals.
    Reference: Ehlers, J.F. "Fisher Transform" Technical Analysis of Stocks & Commodities, 2002.
    """
    hl2 = (high + low) / 2
    
    # Calculate highest high and lowest low over period
    hh = pd.Series(hl2).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(hl2).rolling(window=period, min_periods=period).min().values
    
    # Normalize to 0-1 range with bounds checking
    range_hl = hh - ll
    range_hl = np.where(range_hl < 0.001, 0.001, range_hl)  # avoid division by zero
    normalized = (hl2 - ll) / range_hl
    normalized = np.clip(normalized, 0.001, 0.999)  # bounds for log calculation
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    fisher = np.nan_to_num(fisher, nan=0.0)
    
    # Signal line (previous fisher value)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load daily HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    
    # 4h HMA for trend confirmation
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    # ATR percentile for volatility scaling (reduce size in high vol)
    atr_percentile = pd.Series(atr).rolling(window=100, min_periods=50).apply(
        lambda x: np.percentile(x, 50), raw=True
    ).values
    atr_percentile = np.nan_to_num(atr_percentile, nan=np.median(atr))
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=np.nanmean(volume))
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    HALF_SIZE = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Daily trend filter (major regime)
        daily_bullish = hma_1d_aligned[i] > 0 and close[i] > hma_1d_aligned[i]
        daily_bearish = hma_1d_aligned[i] > 0 and close[i] < hma_1d_aligned[i]
        daily_neutral = abs(close[i] - hma_1d_aligned[i]) < hma_1d_aligned[i] * 0.02 if hma_1d_aligned[i] > 0 else True
        
        # Fisher Transform signals
        fisher_long_cross = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_short_cross = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        fisher_extreme_long = fisher[i] < -1.8 and fisher[i] > fisher_signal[i]
        fisher_extreme_short = fisher[i] > 1.8 and fisher[i] < fisher_signal[i]
        fisher_rising = fisher[i] > fisher_signal[i]
        fisher_falling = fisher[i] < fisher_signal[i]
        
        # 4h HMA trend
        hma_trend_long = hma_21[i] > hma_50[i]
        hma_trend_short = hma_21[i] < hma_50[i]
        
        # RSI momentum filter
        rsi_bullish = rsi[i] > 40 and rsi[i] < 75
        rsi_bearish = rsi[i] > 25 and rsi[i] < 60
        rsi_rising = rsi[i] > rsi[i-3] if i > 3 else True
        rsi_falling = rsi[i] < rsi[i-3] if i > 3 else True
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_sma[i] * 0.9 if vol_sma[i] > 0 else True
        
        # Price position vs HMA21
        price_above_hma = close[i] > hma_21[i]
        price_below_hma = close[i] < hma_21[i]
        
        # Volatility scaling (reduce size when ATR is high)
        vol_scale = 1.0
        if atr_percentile[i] > 0:
            vol_scale = min(1.5, max(0.6, atr_percentile[i] / (atr[i] + 0.0001)))
        vol_scale = np.clip(vol_scale, 0.6, 1.5)
        
        SIZE = BASE_SIZE * vol_scale
        SIZE = np.clip(SIZE, 0.15, 0.35)  # Keep within bounds
        
        # Entry logic - MULTIPLE triggers to ensure trades (Rule 9)
        new_signal = 0.0
        
        # LONG ENTRY TRIGGERS (any one can trigger)
        # Trigger 1: Fisher cross above -1.5 (reversal signal)
        if fisher_long_cross and (daily_bullish or daily_neutral):
            new_signal = SIZE
        # Trigger 2: Fisher extreme long + turning up + RSI ok
        elif fisher_extreme_long and rsi_bullish and price_above_hma:
            new_signal = SIZE
        # Trigger 3: Daily bullish + Fisher rising + HMA trend long
        elif daily_bullish and fisher_rising and hma_trend_long:
            new_signal = SIZE
        # Trigger 4: Fisher rising from oversold + volume confirmation
        elif fisher_rising and fisher[i] < -1.0 and vol_confirm:
            new_signal = SIZE
        # Trigger 5: RSI rising from neutral + Fisher support
        elif rsi_rising and rsi[i] > 45 and fisher[i] > fisher_signal[i]:
            new_signal = SIZE
        
        # SHORT ENTRY TRIGGERS (any one can trigger)
        # Trigger 1: Fisher cross below +1.5 (reversal signal)
        if fisher_short_cross and (daily_bearish or daily_neutral):
            new_signal = -SIZE
        # Trigger 2: Fisher extreme short + turning down + RSI ok
        elif fisher_extreme_short and rsi_bearish and price_below_hma:
            new_signal = -SIZE
        # Trigger 3: Daily bearish + Fisher falling + HMA trend short
        elif daily_bearish and fisher_falling and hma_trend_short:
            new_signal = -SIZE
        # Trigger 4: Fisher falling from overbought + volume confirmation
        elif fisher_falling and fisher[i] > 1.0 and vol_confirm:
            new_signal = -SIZE
        # Trigger 5: RSI falling from neutral + Fisher support
        elif rsi_falling and rsi[i] < 55 and fisher[i] < fisher_signal[i]:
            new_signal = -SIZE
        
        # Stoploss logic (Rule 6) - ATR based with trailing
        if position_side > 0 and entry_price > 0:
            stop_loss = entry_price - 2.5 * entry_atr
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for longs
            else:
                new_trailing = close[i] - 2.5 * atr[i]
                if new_trailing > trailing_stop:
                    trailing_stop = new_trailing
                if close[i] < trailing_stop and trailing_stop > stop_loss:
                    new_signal = 0.0
                # Take partial profit at 2.5R
                elif close[i] > entry_price + 2.5 * entry_atr and signals[i-1] == SIZE:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price > 0:
            stop_loss = entry_price + 2.5 * entry_atr
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for shorts
            else:
                new_trailing = close[i] + 2.5 * atr[i]
                if new_trailing < trailing_stop or trailing_stop == 0:
                    trailing_stop = new_trailing
                if close[i] > trailing_stop and trailing_stop < stop_loss:
                    new_signal = 0.0
                # Take partial profit at 2.5R
                elif close[i] < entry_price - 2.5 * entry_atr and signals[i-1] == -SIZE:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            entry_atr = atr[i]
            trailing_stop = close[i] - 2.5 * entry_atr if position_side > 0 else close[i] + 2.5 * entry_atr
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price = close[i]
                position_side = np.sign(new_signal)
                entry_atr = atr[i]
                trailing_stop = close[i] - 2.5 * entry_atr if position_side > 0 else close[i] + 2.5 * entry_atr
        elif new_signal == 0 and position_side != 0:
            position_side = 0
            entry_price = 0.0
            entry_atr = 0.0
            trailing_stop = 0.0
        
        signals[i] = new_signal
    
    return signals