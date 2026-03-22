#!/usr/bin/env python3
"""
Experiment #246: 1d HMA Trend + RSI Pullback + ADX Filter + 1w HTF Bias

Hypothesis: Daily timeframe captures major trend moves while filtering intraday noise.
Using HMA for smoother trend detection + RSI(7) pullback entries + ADX(14) trend strength
filter + 1w HMA for macro bias. This is simpler than failed complex regime strategies.

Why this might work on 1d:
- 1d has less noise than 1h/4h, better for trend following
- HMA(21) smoother than EMA with less lag
- RSI(7) pullbacks in trend = high probability entries (not extremes)
- ADX(14)>20 filters out choppy ranges
- 1w HMA provides macro trend confirmation
- Conservative sizing (0.30) + 2*ATR stoploss controls drawdown

Key differences from failed experiments:
- Simpler logic (fewer conflicting filters = more trades)
- RSI pullback (35-45 long, 55-65 short) not extremes (20/80)
- ADX threshold 20 (not 25+) for more signals
- Discrete position sizing to minimize fee churn
- MUST generate ≥10 trades on train, ≥3 on test

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.30 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_pullback_adx_1w_hma_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=7):
    """Calculate RSI (Relative Strength Index) - faster period for pullbacks."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate +DM and -DM
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Calculate TR
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smooth with Wilder's method (EMA with span=period)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

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
    hma_21 = calculate_hma(close, 21)
    rsi_7 = calculate_rsi(close, 7)
    adx_14 = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(rsi_7[i]) or np.isnan(adx_14[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 1w HMA = macro trend bias
        bull_macro = close[i] > hma_1w_aligned[i]
        bear_macro = close[i] < hma_1w_aligned[i]
        
        # === TREND DETECTION ===
        # Price vs HMA(21)
        bull_trend = close[i] > hma_21[i]
        bear_trend = close[i] < hma_21[i]
        
        # HMA slope (3-bar lookback for responsiveness)
        hma_slope_bullish = hma_21[i] > hma_21[i-3] if i >= 3 else False
        hma_slope_bearish = hma_21[i] < hma_21[i-3] if i >= 3 else False
        
        # === TREND STRENGTH ===
        # ADX > 20 = trending market (not choppy)
        is_trending = adx_14[i] > 20
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        # --- LONG ENTRY: Trend + Pullback + Macro Bias ---
        # Conditions:
        # 1. Price above HMA(21) = uptrend
        # 2. HMA sloping up
        # 3. RSI(7) pullback to 35-50 (not overbought, not extreme oversold)
        # 4. ADX > 20 = trending (not choppy)
        # 5. 1w HMA bullish OR neutral (not strongly bearish)
        if bull_trend and hma_slope_bullish and is_trending:
            if 35 <= rsi_7[i] <= 50:  # Pullback zone
                if bull_macro:  # Macro bias supportive
                    new_signal = SIZE_ENTRY
                elif not bear_macro:  # At least not bearish macro
                    new_signal = SIZE_ENTRY * 0.7  # Reduced size
        
        # --- SHORT ENTRY: Trend + Pullback + Macro Bias ---
        # Conditions:
        # 1. Price below HMA(21) = downtrend
        # 2. HMA sloping down
        # 3. RSI(7) pullback to 50-65 (not oversold, not extreme overbought)
        # 4. ADX > 20 = trending (not choppy)
        # 5. 1w HMA bearish OR neutral (not strongly bullish)
        if bear_trend and hma_slope_bearish and is_trending:
            if 50 <= rsi_7[i] <= 65:  # Pullback zone
                if bear_macro:  # Macro bias supportive
                    new_signal = -SIZE_ENTRY
                elif not bull_macro:  # At least not bullish macro
                    new_signal = -SIZE_ENTRY * 0.7  # Reduced size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.0 * ATR below highest close
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.0 * ATR above lowest close
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TAKE PROFIT: Reduce to half at 2R profit ===
        if in_position and new_signal != 0.0 and position_side != 0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * entry_atr:
                    new_signal = SIZE_HALF  # Take partial profit
            if position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * entry_atr:
                    new_signal = -SIZE_HALF  # Take partial profit
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly reduced size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals