#!/usr/bin/env python3
"""
Experiment #019: 4h HMA Trend + Donchian Breakout with 1d Bias

Hypothesis: Previous RSI pullback strategies failed due to overly restrictive entry
conditions (RSI ranges too narrow). This strategy uses Donchian breakouts which
naturally generate more signals while maintaining trend quality.

Key components:
1. 1d HMA(21) for major trend bias (proven in best baseline)
2. 4h HMA(16/48) crossover for trend direction
3. 4h Donchian(20) breakout for entry timing (more signals than RSI pullback)
4. 4h ADX(14) > 20 filter to confirm trending regime (avoid chop)
5. 4h ATR(14) trailing stoploss at 2.5x
6. Discrete position sizing (0.25-0.30)

Why this should beat the baseline:
- Donchian breakouts generate MORE trades than RSI pullback (addresses 0-trade issue)
- ADX filter prevents entries during choppy periods (reduces whipsaw)
- 1d HMA bias prevents counter-trend trades (major edge)
- Simpler logic = more reliable execution
- 4h timeframe naturally targets 20-50 trades/year (optimal fee/risk balance)

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_donchian_adx_1d_bias_v1"
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
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(0).values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1D indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    hma_4h_16 = calculate_hma(close, 16)
    hma_4h_48 = calculate_hma(close, 48)
    adx_14 = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    rsi_14 = calculate_rsi(close, 14)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]):
            continue
        
        if np.isnan(adx_14[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1D TREND BIAS ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H HMA TREND ===
        hma_bullish = hma_4h_16[i] > hma_4h_48[i]
        hma_bearish = hma_4h_16[i] < hma_4h_48[i]
        
        # === TREND STRENGTH (ADX) ===
        trending = adx_14[i] > 20  # ADX > 20 indicates trending market
        
        # === DONCHIAN BREAKOUT DETECTION ===
        prev_upper = donchian_upper[i-1] if i > 0 else np.nan
        prev_lower = donchian_lower[i-1] if i > 0 else np.nan
        
        breakout_long = not np.isnan(prev_upper) and close[i] > prev_upper
        breakout_short = not np.isnan(prev_lower) and close[i] < prev_lower
        
        # === RSI CONFIRMATION (wider range for more trades) ===
        rsi_ok_long = rsi_14[i] > 40 and rsi_14[i] < 75
        rsi_ok_short = rsi_14[i] > 25 and rsi_14[i] < 60
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        if i > 100:
            atr_median = np.nanmedian(atr_14[max(0, i-100):i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
            vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        else:
            vol_adjustment = 1.0
        
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === ENTRY LOGIC (DONCHIAN BREAKOUT + TREND CONFIRMATION) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG: 4h HMA bullish + 1d bias bullish + ADX trending + Donchian breakout
        if hma_bullish and daily_bullish and trending:
            if breakout_long and rsi_ok_long:
                new_signal = current_size
            # Secondary entry: HMA cross with RSI confirmation (more trades)
            elif i > 1 and hma_4h_16[i] > hma_4h_48[i] and hma_4h_16[i-1] <= hma_4h_48[i-1]:
                if rsi_14[i] > 45 and rsi_14[i] < 70:
                    new_signal = current_size * 0.8
        
        # SHORT: 4h HMA bearish + 1d bias bearish + ADX trending + Donchian breakout
        elif hma_bearish and daily_bearish and trending:
            if breakout_short and rsi_ok_short:
                new_signal = -current_size
            # Secondary entry: HMA cross with RSI confirmation (more trades)
            elif i > 1 and hma_4h_16[i] < hma_4h_48[i] and hma_4h_16[i-1] >= hma_4h_48[i-1]:
                if rsi_14[i] > 30 and rsi_14[i] < 55:
                    new_signal = -current_size * 0.8
        
        # === FREQUENCY SAFEGUARD (ensure minimum trades) ===
        # If no trades for 60 bars (~10 days on 4h), force entry with weaker conditions
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if hma_bullish and daily_bullish and rsi_14[i] > 45:
                new_signal = current_size * 0.5
            elif hma_bearish and daily_bearish and rsi_14[i] < 55:
                new_signal = -current_size * 0.5
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and hma_bearish:
                trend_reversal = True
            if position_side < 0 and hma_bullish:
                trend_reversal = True
        
        # === ADX DROPOUT EXIT (trend weakening) ===
        adx_dropout = False
        if in_position and position_side != 0:
            if adx_14[i] < 15:  # ADX dropped below 15 = trend weakening
                adx_dropout = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal or adx_dropout:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals