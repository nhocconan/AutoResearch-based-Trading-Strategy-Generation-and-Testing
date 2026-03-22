#!/usr/bin/env python3
"""
Experiment #259: 15m RSI Mean Reversion with 4h/1h HMA Trend Filter

Hypothesis: 15m timeframe is noisy for trend following but excellent for mean reversion.
Using 4h HMA for primary trend bias + 1h HMA for intermediate confirmation.
Entry on RSI(7) extremes (oversold <25 in uptrend, overbought >75 in downtrend).
This is LOOSER than previous attempts to ensure trades happen.

Why this might work:
- 15m RSI extremes happen frequently (ensures trades)
- 4h HMA filter prevents counter-trend trades in strong trends
- 1h HMA adds intermediate confirmation without being too restrictive
- Mean reversion works well in 2025 bear/range market
- Conservative sizing (0.25) + ATR stoploss controls drawdown

Key improvements over failed 15m experiments:
- #247 (15m chop regime): Sharpe=-3.27 - too many filters, 0 trades likely
- #253 (15m trend pullback): Sharpe=-5.51 - trend following fails on 15m noise
- This uses SIMPLE RSI extremes + HTF trend = more trades, less whipsaw
- RSI(7) is faster than RSI(14), catches more reversals
- Entry threshold RSI<25/>75 is looser than RSI<20/>80

Timeframe: 15m (REQUIRED for this experiment)
HTF: 4h and 1h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_meanrev_4h_1h_hma_atr_v1"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger(close, 20, 2.0)
    ema_50 = calculate_ema(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.125
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_price_idx = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME TREND BIAS ===
        # 4h HMA = primary trend direction
        # 1h HMA = intermediate confirmation
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        bull_trend_1h = close[i] > hma_1h_aligned[i]
        bear_trend_1h = close[i] < hma_1h_aligned[i]
        
        # === RSI MEAN REVERSION SIGNALS ===
        # Long: RSI(7) oversold in uptrend
        rsi_oversold = rsi_7[i] < 25
        rsi_overbought = rsi_7[i] > 75
        
        # Bollinger Band confirmation (price near lower band for long, upper for short)
        bb_long_conf = close[i] < bb_lower[i] * 1.005  # at or below lower band
        bb_short_conf = close[i] > bb_upper[i] * 0.995  # at or above upper band
        
        # === ENTRY SIGNALS (LOOSE CONDITIONS TO ENSURE TRADES) ===
        new_signal = 0.0
        
        # Long entry: RSI oversold + 4h uptrend (1h optional confirmation)
        # Need EITHER 4h bull OR 1h bull (not both - too restrictive)
        if rsi_oversold and (bull_trend_4h or bull_trend_1h):
            new_signal = SIZE_BASE
        
        # Short entry: RSI overbought + 4h downtrend (1h optional confirmation)
        if rsi_overbought and (bear_trend_4h or bear_trend_1h):
            new_signal = -SIZE_BASE
        
        # Additional mean reversion: RSI extreme + BB touch (works in range markets)
        # This catches reversals when HTF trend is unclear
        if rsi_7[i] < 20 and bb_long_conf and close[i] > ema_50[i]:
            new_signal = SIZE_BASE  # Strong oversold + BB + above EMA50
        
        if rsi_7[i] > 80 and bb_short_conf and close[i] < ema_50[i]:
            new_signal = -SIZE_BASE  # Strong overbought + BB + below EMA50
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
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
        
        # === TAKE PROFIT: Reduce to half at 2R profit ===
        if in_position and new_signal != 0.0 and position_side != 0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * atr[entry_price_idx]:
                    new_signal = SIZE_HALF  # Take partial profit
            if position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * atr[entry_price_idx]:
                    new_signal = -SIZE_HALF  # Take partial profit
        
        # === EXIT ON RSI REVERSAL ===
        # Exit long if RSI goes overbought, exit short if RSI goes oversold
        if in_position and new_signal != 0.0:
            if position_side > 0 and rsi_7[i] > 70:
                new_signal = SIZE_HALF  # Take profit on RSI reversal
            if position_side < 0 and rsi_7[i] < 30:
                new_signal = -SIZE_HALF  # Take profit on RSI reversal
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_price_idx = i
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_price_idx = i
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly adjusted size)
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