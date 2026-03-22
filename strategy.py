#!/usr/bin/env python3
"""
Experiment #008: 30m Multi-Timeframe Confluence with Session Filter

Hypothesis: After 7 consecutive failures with complex regime-switching strategies,
the issue is OVER-FITTING. This strategy uses SIMPLE but STRICT confluence:

1. DUAL HTF TREND BIAS: Both 4h AND 1d HMA must agree (proven stable filter)
2. 30m RSI PULLBACK: Enter on pullback within HTF trend (RSI 35-45 long, 55-65 short)
3. VOLUME CONFIRMATION: Volume > 0.8 * 20-bar SMA (filters fakeouts)
4. SESSION FILTER: Only trade 8-20 UTC (highest liquidity, lowest slippage)
5. ATR TRAILING STOP: 2.5 * ATR(14) to protect capital

Why this should work when #001-#007 failed:
- SIMPLER logic = less overfitting (failed strategies had 3+ regime modes)
- DUAL HTF filter = more stable than single 4h HMA
- SESSION filter = critical for 30m (avoid Asian session whipsaw)
- RSI pullback = proven edge from baseline strategy (Sharpe=5.4)
- Strict entry = ensures 30-80 trades/year target (not 200+)

Timeframe: 30m (REQUIRED)
HTF: 4h + 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete (conservative for lower TF)
Stoploss: 2.5 * ATR(14) trailing via signal→0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_pullback_4h_1d_hma_session_vol_v1"
timeframe = "30m"
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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

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
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        if np.isnan(vol_sma[i]) or vol_sma[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        # open_time is in milliseconds since epoch
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour_utc <= 20
        
        # === DUAL HTF TREND BIAS (4h + 1d must agree) ===
        bull_bias_4h = close[i] > hma_4h_aligned[i]
        bear_bias_4h = close[i] < hma_4h_aligned[i]
        bull_bias_1d = close[i] > hma_1d_aligned[i]
        bear_bias_1d = close[i] < hma_1d_aligned[i]
        
        # Both HTF must agree for strong bias
        strong_bull_bias = bull_bias_4h and bull_bias_1d
        strong_bear_bias = bear_bias_4h and bear_bias_1d
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_sma[i]
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI pulled back to 35-45 within bullish HTF trend
        rsi_pullback_long = 35 <= rsi_14[i] <= 48
        # Short: RSI pulled back to 52-65 within bearish HTF trend
        rsi_pullback_short = 52 <= rsi_14[i] <= 65
        
        # === ATR-BASED POSITION SIZING ===
        # Reduce size when volatility is high (protect in crashes)
        if i > 100:
            atr_median = np.nanmedian(atr_14[100:i])
            if atr_median > 0:
                atr_ratio = atr_14[i] / atr_median
                atr_ratio = np.clip(atr_ratio, 0.5, 2.0)
                size_multiplier = 1.0 / atr_ratio
                current_size = BASE_SIZE * size_multiplier
                current_size = np.clip(current_size, 0.15, 0.35)
            else:
                current_size = BASE_SIZE
        else:
            current_size = BASE_SIZE
        
        # === ENTRY LOGIC (ALL conditions must be met) ===
        new_signal = 0.0
        
        # LONG: Strong bull bias + RSI pullback + volume + session
        if strong_bull_bias and rsi_pullback_long and volume_confirmed and in_session:
            new_signal = current_size
        
        # SHORT: Strong bear bias + RSI pullback + volume + session
        elif strong_bear_bias and rsi_pullback_short and volume_confirmed and in_session:
            new_signal = -current_size
        
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
            # Exit long if BOTH HTF turn bearish
            if position_side > 0 and strong_bear_bias:
                trend_reversal = True
            # Exit short if BOTH HTF turn bullish
            if position_side < 0 and strong_bull_bias:
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
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals