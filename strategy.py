#!/usr/bin/env python3
"""
Experiment #196: 4h Fisher Transform + 1d HMA Trend + Volume Confirmation + ATR Stop

Hypothesis: Fisher Transform excels at catching reversals in bear/range markets (2025+),
while 1d HMA provides stable higher-timeframe bias to avoid counter-trend trades.
Volume confirmation filters false breakouts. This addresses the failure of pure
trend-following strategies in recent experiments.

Why this might work:
- Fisher Transform (period=9) is proven for reversal detection in non-trending markets
- 1d HMA filter prevents trading against the major trend
- Volume confirmation (>SMA20) ensures genuine momentum behind moves
- 4h timeframe balances signal frequency with noise reduction
- Conservative sizing (0.25) controls drawdown in volatile periods

Learning from failures:
- #190 (4h Fisher + vol spike): Sharpe=0.000 - likely 0 trades, conditions too strict
- #184-195: All negative Sharpe - pure trend or pure mean-revert both failing
- Need regime-adaptive OR reversal-focused approach for 2025 bear market
- Fisher Transform specifically mentioned as working in "bear rallies"

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_1d_hma_volume_atr_v2"
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

def calculate_fisher(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5 (oversold reversal)
    Short when Fisher crosses below +1.5 (overbought reversal)
    """
    close_s = pd.Series(close)
    
    # Calculate highest high and lowest low over period
    hh = close_s.rolling(window=period, min_periods=period).max()
    ll = close_s.rolling(window=period, min_periods=period).min()
    
    # Normalize price to range -1 to +1
    x = (2 * close - (hh + ll)) / (hh - ll + 1e-10)
    x = x.clip(-0.999, 0.999)  # Prevent log domain errors
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + x) / (1 - x + 1e-10))
    
    # Signal line (1-period lag of fisher)
    fisher_signal = fisher.shift(1)
    
    return fisher.values, fisher_signal.values

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
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_volume_sma(volume, period=20):
    """Calculate SMA of volume for volume confirmation."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher(close, 9)
    rsi = calculate_rsi(close, 14)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = higher timeframe trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_long_cross = (fisher_signal[i] < -1.5) and (fisher[i] >= -1.5)
        fisher_short_cross = (fisher_signal[i] > 1.5) and (fisher[i] <= 1.5)
        
        # Also allow continuation signals (Fisher moving in trend direction)
        fisher_bullish = fisher[i] > fisher_signal[i]  # Fisher rising
        fisher_bearish = fisher[i] < fisher_signal[i]  # Fisher falling
        
        # === VOLUME CONFIRMATION ===
        # Volume must be above 20-period SMA for signal validity
        volume_confirmed = volume[i] > vol_sma[i]
        
        # === RSI FILTER ===
        # Avoid extreme overbought/oversold counter-trend entries
        rsi_not_extreme_long = rsi[i] < 80  # Don't long at extreme overbought
        rsi_not_extreme_short = rsi[i] > 20  # Don't short at extreme oversold
        
        # === EMA STRUCTURE ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 1d bullish + (Fisher cross OR Fisher rising) + volume + RSI filter
        # More flexible to ensure enough trades
        if bull_trend_1d:
            if (fisher_long_cross or (fisher_bullish and rsi[i] < 60)):
                if volume_confirmed and rsi_not_extreme_long:
                    new_signal = SIZE_BASE
        
        # Short: 1d bearish + (Fisher cross OR Fisher falling) + volume + RSI filter
        if bear_trend_1d:
            if (fisher_short_cross or (fisher_bearish and rsi[i] > 40)):
                if volume_confirmed and rsi_not_extreme_short:
                    new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals