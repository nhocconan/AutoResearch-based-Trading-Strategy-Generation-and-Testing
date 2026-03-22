#!/usr/bin/env python3
"""
Experiment #015: 1h Multi-Timeframe Pullback Strategy with Session + Volume Filter

Hypothesis: Lower TF (1h) strategies fail due to too many trades and fee drag.
This strategy uses STRICT confluence to limit trades to 30-80/year:
1. 4h HMA trend direction (major trend)
2. 1d HMA bias filter (confirm major trend)
3. 1h RSI pullback (35-45 for long, 55-65 for short - NOT extremes)
4. Volume > 0.8x 20-bar average (confirm participation)
5. Session filter (8-20 UTC only - high liquidity hours)
6. ATR trailing stop (2.5x ATR)

Why this should work:
- 4h + 1d alignment filters out counter-trend noise
- Moderate RSI pullback (not extreme) catches continuation, not reversals
- Session filter avoids Asian session noise (low liquidity whipsaws)
- Volume confirmation ensures real moves, not fakeouts
- 1h entry within 4h trend = HTF frequency with LTF precision

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h + 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete (conservative for lower TF)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h1d_session_vol_v1"
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

def calculate_volume_ma(volume, period=20):
    """Calculate moving average of volume."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_ma

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
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
    
    # Calculate 4H indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, 50)
    
    # Calculate 1D indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_ma_20 = calculate_volume_ma(volume, 20)
    
    # Extract UTC hours for session filter
    utc_hours = np.array([get_utc_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(vol_ma_20[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        current_hour = utc_hours[i]
        in_session = 8 <= current_hour <= 20
        
        # === VOLUME FILTER ===
        vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        volume_confirmed = vol_ratio >= 0.8
        
        # === 4H TREND (major trend direction) ===
        hma_4h_bullish = hma_4h_21_aligned[i] > hma_4h_50_aligned[i]
        hma_4h_bearish = hma_4h_21_aligned[i] < hma_4h_50_aligned[i]
        
        # Price relative to 4h HMA
        price_above_hma4h = close[i] > hma_4h_21_aligned[i]
        price_below_hma4h = close[i] < hma_4h_21_aligned[i]
        
        # === 1D BIAS (confirm major trend) ===
        hma_1d_bullish = close[i] > hma_1d_21_aligned[i]
        hma_1d_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 1H RSI PULLBACK (moderate, not extreme) ===
        rsi_oversold_pullback = 35 <= rsi_14[i] <= 48
        rsi_overbought_pullback = 52 <= rsi_14[i] <= 65
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        atr_ratio = atr_14[i] / np.nanmedian(atr_14[max(0, i-100):i]) if i > 100 else 1.0
        vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        # Round to discrete levels
        if current_size < 0.22:
            current_size = 0.20
        elif current_size < 0.28:
            current_size = 0.25
        else:
            current_size = 0.30
        
        # === ENTRY LOGIC (3+ CONFLUENCE REQUIRED) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: 4h bullish + 1d bullish + RSI pullback + session + volume
        long_confluence = (
            hma_4h_bullish and
            hma_1d_bullish and
            price_above_hma4h and
            rsi_oversold_pullback and
            in_session and
            volume_confirmed
        )
        
        # SHORT ENTRY: 4h bearish + 1d bearish + RSI pullback + session + volume
        short_confluence = (
            hma_4h_bearish and
            hma_1d_bearish and
            price_below_hma4h and
            rsi_overbought_pullback and
            in_session and
            volume_confirmed
        )
        
        if long_confluence:
            new_signal = current_size
        elif short_confluence:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 100 bars (~4 days on 1h), allow weaker entry
        if bars_since_last_trade > 100 and new_signal == 0.0 and not in_position:
            weak_long = (
                hma_4h_bullish and
                hma_1d_bullish and
                rsi_14[i] < 45 and
                in_session
            )
            weak_short = (
                hma_4h_bearish and
                hma_1d_bearish and
                rsi_14[i] > 55 and
                in_session
            )
            if weak_long:
                new_signal = current_size * 0.7
            elif weak_short:
                new_signal = -current_size * 0.7
        
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
            if position_side > 0 and hma_4h_bearish:
                trend_reversal = True
            if position_side < 0 and hma_4h_bullish:
                trend_reversal = True
        
        # === RSI EXTREME EXIT (take profit) ===
        rsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and rsi_14[i] > 75:
                rsi_exit = True
            if position_side < 0 and rsi_14[i] < 25:
                rsi_exit = True
        
        # Apply stoploss or trend reversal or RSI exit
        if stoploss_triggered or trend_reversal or rsi_exit:
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