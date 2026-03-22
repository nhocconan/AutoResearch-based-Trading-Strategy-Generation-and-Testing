#!/usr/bin/env python3
"""
Experiment #005: 1h Fisher Transform + 4h HMA Trend + Volume/Session Filter

Hypothesis: Previous regime-based strategies (CHOP + Connors RSI) failed because:
1. Regime detection is too lagging (CHOP looks back 14 bars)
2. Too many confluence filters kill trade frequency
3. RSI extremes don't work well in bear/range markets (2025 test period)

New approach uses:
1. Fisher Transform (period=9): Better at catching reversals than RSI, especially in bear markets
   - Long when Fisher crosses above -1.5 (oversold reversal)
   - Short when Fisher crosses below +1.5 (overbought reversal)
2. 4h HMA(21) for trend direction: Price above = long bias, below = short bias
3. 1d HMA(21) for major trend filter: Only take trades in direction of daily trend
4. Volume filter: Volume > 0.8x 20-bar average (filters fake breakouts)
5. Session filter: Only 8-20 UTC (high liquidity hours, reduces false signals)
6. ATR-based position sizing: Smaller size when volatility is high

Why this should work:
- Fisher Transform has proven edge in mean-reversion during bear markets
- 4h trend filter reduces whipsaw on 1h timeframe
- Session filter cuts trades by ~50% but improves win rate
- Volume confirmation filters low-liquidity fakeouts
- Discrete position sizes minimize fee churn

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 volatility-adjusted, discrete levels
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year (session filter helps achieve this)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_4h_hma_volume_session_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Converts price to Gaussian normal distribution for better reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    # Calculate typical price
    typical = (high + low) / 2.0
    typical_s = pd.Series(typical)
    
    # Calculate EMA of typical price
    ema_typical = typical_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Calculate Fisher input (normalized price)
    # Use (typical - ema) / (0.001 * ema) to avoid division by zero
    fisher_input = (typical - ema_typical.values) / (0.001 * ema_typical.values + 1e-10)
    fisher_input = np.clip(fisher_input, -1.0, 1.0)  # Clamp to valid range
    
    # Apply Fisher transform formula
    fisher = 0.5 * np.log((1 + fisher_input) / (1 - fisher_input + 1e-10))
    
    # Calculate Fisher signal line (1-bar lag)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_volume_ratio(volume, period=20):
    """Calculate volume relative to 20-bar average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = volume / vol_avg.values
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    dt = pd.to_datetime(open_time, unit='ms', utc=True)
    return dt.hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # Extract UTC hours for session filter
    utc_hours = np.array([get_utc_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    fisher_cross_long = False
    fisher_cross_short = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= utc_hours[i] <= 20
        
        # === VOLUME FILTER ===
        volume_ok = vol_ratio[i] >= 0.8  # Volume at least 80% of 20-bar avg
        
        # === HTF TREND BIAS ===
        # 4h trend: price above HMA = bullish bias
        hma_4h_bullish = close[i] > hma_4h_21_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # 1d trend: major trend confirmation
        hma_1d_bullish = close[i] > hma_1d_21_aligned[i]
        hma_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Detect crosses (current bar crosses threshold, previous bar did not)
        # Long: Fisher crosses above -1.5
        fisher_cross_long = (fisher[i] > -1.5) and (fisher_signal[i] <= -1.5)
        # Short: Fisher crosses below +1.5
        fisher_cross_short = (fisher[i] < 1.5) and (fisher_signal[i] >= 1.5)
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        atr_ratio = atr_14[i] / np.nanmedian(atr_14[max(0, i-100):i]) if i > 100 else 1.0
        vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        # Round to discrete levels
        if current_size < 0.25:
            current_size = 0.20
        else:
            current_size = 0.30
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Need Fisher cross + 4h bullish + volume + session
        # 1d trend is bonus (not required but preferred)
        if fisher_cross_long and in_session and volume_ok:
            # Require 4h bullish bias
            if hma_4h_bullish:
                # 1d bullish = full size, 1d bearish = half size (counter-trend)
                if hma_1d_bullish:
                    new_signal = current_size
                else:
                    new_signal = current_size * 0.5  # Smaller for counter-trend
        
        # SHORT ENTRY: Need Fisher cross + 4h bearish + volume + session
        if fisher_cross_short and in_session and volume_ok:
            # Require 4h bearish bias
            if hma_4h_bearish:
                # 1d bearish = full size, 1d bullish = half size (counter-trend)
                if hma_1d_bearish:
                    new_signal = -current_size
                else:
                    new_signal = -current_size * 0.5  # Smaller for counter-trend
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 80 bars (~5 days on 1h), allow entry without session filter
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            if fisher_cross_long and volume_ok and hma_4h_bullish:
                new_signal = current_size * 0.6  # Smaller size for relaxed conditions
            elif fisher_cross_short and volume_ok and hma_4h_bearish:
                new_signal = -current_size * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 4h HMA turns bearish
            if position_side > 0 and hma_4h_bearish:
                trend_reversal = True
            # Exit short if 4h HMA turns bullish
            if position_side < 0 and hma_4h_bullish:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals