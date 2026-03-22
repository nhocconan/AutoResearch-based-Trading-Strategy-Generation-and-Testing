#!/usr/bin/env python3
"""
Experiment #018: 30m RSI Pullback + 4h/1d HMA Trend + Volume Filter

Hypothesis: Lower timeframe (30m) with HTF trend filter can beat 4h strategies
by catching more entry opportunities within the HTF trend. Key improvements:
1. 4h HMA(21) for primary trend direction (not 12h - too slow)
2. 1d HMA(50) for major bias filter (avoid counter-trend in strong moves)
3. 30m RSI(14) pullback entries (RSI 30-45 long, 55-70 short) - NOT extremes
4. Volume confirmation (1.2x 20-bar median) to filter false breakouts
5. Session filter (8-20 UTC) for liquid hours only
6. ATR(14) stoploss at 2.5x for risk management

Why this should work:
- RSI pullback (not extremes) generates MORE trades than Connors RSI
- 4h trend filter prevents counter-trend trades (main failure of lower TF)
- Volume filter reduces false signals in low liquidity
- 30m TF targets 40-80 trades/year (fee-efficient for this timeframe)
- Discrete sizing (0.25) minimizes fee churn

Timeframe: 30m (REQUIRED)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_pullback_4h_1d_hma_volume_v1"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_avg = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    loss_avg = np.where(loss_avg == 0, 1e-10, loss_avg)
    rs = gain_avg / loss_avg
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
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

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio: current volume / median volume over period."""
    vol_s = pd.Series(volume)
    vol_median = vol_s.rolling(window=period, min_periods=period).median().values
    vol_ratio = volume / vol_median
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    return (open_time // 3600000) % 24

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
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 1d indicators
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    
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
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        # === HTF TREND BIAS (4h + 1d) ===
        # 4h HMA for primary trend
        htf_4h_bullish = close[i] > hma_4h_21_aligned[i]
        htf_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # 1d HMA for major bias (avoid strong counter-trend)
        htf_1d_bullish = close[i] > hma_1d_50_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_50_aligned[i]
        
        # === 30m RSI PULLBACK (NOT extremes - more trades) ===
        # Long: RSI 30-45 (pullback in uptrend, not oversold crash)
        # Short: RSI 55-70 (pullback in downtrend, not overbought spike)
        rsi_long_pullback = 30 <= rsi_14[i] <= 48
        rsi_short_pullback = 52 <= rsi_14[i] <= 70
        
        # === VOLUME CONFIRMATION ===
        volume_ok = vol_ratio[i] >= 1.15  # 15% above median
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        if i > 100:
            atr_median = np.nanmedian(atr_14[max(0, i-100):i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
            vol_adjustment = np.clip(1.0 / atr_ratio, 0.8, 1.2)
        else:
            vol_adjustment = 1.0
        
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.30)
        # Round to discrete levels
        if current_size > 0.27:
            current_size = 0.30
        elif current_size > 0.17:
            current_size = 0.25
        else:
            current_size = 0.20
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: 4h bullish + 1d not strongly bearish + RSI pullback + volume + session
        if htf_4h_bullish and not htf_1d_bearish and rsi_long_pullback:
            if in_session and volume_ok:
                new_signal = current_size
            elif in_position and position_side > 0:
                # Hold position even outside session
                new_signal = current_size
        
        # SHORT ENTRY: 4h bearish + 1d not strongly bullish + RSI pullback + volume + session
        elif htf_4h_bearish and not htf_1d_bullish and rsi_short_pullback:
            if in_session and volume_ok:
                new_signal = -current_size
            elif in_position and position_side < 0:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 40 bars (~20 hours on 30m), allow weaker entry
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            if htf_4h_bullish and rsi_14[i] < 50:
                new_signal = current_size * 0.8
            elif htf_4h_bearish and rsi_14[i] > 50:
                new_signal = -current_size * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
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
            # Exit long if 4h HMA turns bearish
            if position_side > 0 and htf_4h_bearish:
                trend_reversal = True
            # Exit short if 4h HMA turns bullish
            if position_side < 0 and htf_4h_bullish:
                trend_reversal = True
            
            # Exit if RSI goes extreme (overbought long, oversold short)
            if position_side > 0 and rsi_14[i] > 75:
                trend_reversal = True
            if position_side < 0 and rsi_14[i] < 25:
                trend_reversal = True
        
        # Apply stoploss or reversals
        if stoploss_triggered or trend_reversal:
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