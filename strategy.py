#!/usr/bin/env python3
"""
Experiment #028: 30m Multi-Timeframe Trend-Pullback with Session Filter

Hypothesis: Lower TF (30m) strategies fail due to too many trades → fee drag.
Solution: Use 4h/1d for TREND DIRECTION, 30m only for ENTRY TIMING.
This gives HTF trade frequency with lower TF execution precision.

Key filters (ALL must align):
1. 4h HMA(21) trend direction (primary signal)
2. 1d HMA(21) confirms major bias (price above for long, below for short)
3. 30m RSI(14) pullback to 40-50 zone (entry timing within HTF trend)
4. Volume > 1.2x 20-bar average (confirmation)
5. Session filter: 8-20 UTC only (high liquidity, avoid Asia overnight)
6. ATR(14) stoploss: 2.5x trailing

Why this might work:
- 4h/1d trend filter reduces trade frequency to ~40-60/year
- RSI pullback ensures we enter on retracement, not chase
- Session filter avoids low-liquidity whipsaw
- Small position size (0.22) controls drawdown

Timeframe: 30m (REQUIRED)
HTF: 4h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.25 (smaller for lower TF)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_trend_rsi_pullback_session_vol_atr_v1"
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
    
    rs = avg_gain / avg_loss
    rs = rs.replace(np.inf, 0).replace(-np.inf, 0).fillna(0)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / vol_avg
    return vol_ratio.values

def get_hour_from_open_time(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

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
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, smaller for lower TF)
    BASE_SIZE = 0.22
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100  # Track last trade for frequency control
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(vol_ratio[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        current_hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= current_hour <= 20
        
        # === 1D TREND BIAS (major trend) ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === 4H TREND DIRECTION (primary signal) ===
        hma_4h_bullish = close[i] > hma_4h_21_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_21_aligned[i]
        
        # === 4H HMA SLOPE (trend momentum) ===
        hma_4h_slope_long = hma_4h_21_aligned[i] > hma_4h_21_aligned[i-1] if i > 0 else False
        hma_4h_slope_short = hma_4h_21_aligned[i] < hma_4h_21_aligned[i-1] if i > 0 else False
        
        # === 30M RSI PULLBACK (entry timing) ===
        # Long: RSI pulled back to 40-50 zone in uptrend
        rsi_pullback_long = 40 <= rsi_14[i] <= 55
        # Short: RSI rallied to 45-60 zone in downtrend
        rsi_pullback_short = 45 <= rsi_14[i] <= 60
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 1.15  # Volume 15% above average
        
        # === VOLATILITY FILTER (avoid extreme vol) ===
        atr_ratio = atr_14[i] / np.nanmedian(atr_14[max(0, i-100):i]) if i > 100 else 1.0
        vol_not_extreme = atr_ratio < 2.5  # Avoid entry during vol spikes
        
        # === POSITION SIZING (volatility-adjusted) ===
        vol_adjustment = np.clip(1.0 / max(0.5, atr_ratio), 0.7, 1.2)
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.18, 0.28)  # Keep in safe range
        
        # === ENTRY LOGIC (ALL conditions must align) ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRY: Need ALL of these
        # 1. In session (8-20 UTC)
        # 2. 1D bullish (major trend up)
        # 3. 4H bullish (primary trend up)
        # 4. 4H HMA sloping up (momentum)
        # 5. RSI pullback to 40-55 (entry timing)
        # 6. Volume confirmed
        # 7. Not extreme volatility
        long_conditions = (
            in_session and
            daily_bullish and
            hma_4h_bullish and
            hma_4h_slope_long and
            rsi_pullback_long and
            volume_confirmed and
            vol_not_extreme
        )
        
        if long_conditions and bars_since_last_trade > 20:
            new_signal = current_size
        
        # SHORT ENTRY: Need ALL of these
        short_conditions = (
            in_session and
            daily_bearish and
            hma_4h_bearish and
            hma_4h_slope_short and
            rsi_pullback_short and
            volume_confirmed and
            vol_not_extreme
        )
        
        if short_conditions and bars_since_last_trade > 20:
            new_signal = -current_size
        
        # === MINIMUM TRADE FREQUENCY SAFEGUARD ===
        # If no trades for 80 bars (~40 hours on 30m), relax one condition
        if bars_since_last_trade > 80 and new_signal == 0.0 and not in_position:
            # Relax volume requirement
            long_relaxed = (
                in_session and
                daily_bullish and
                hma_4h_bullish and
                rsi_pullback_long and
                vol_not_extreme
            )
            short_relaxed = (
                in_session and
                daily_bearish and
                hma_4h_bearish and
                rsi_pullback_short and
                vol_not_extreme
            )
            
            if long_relaxed:
                new_signal = current_size * 0.7  # Smaller size for relaxed entry
            elif short_relaxed:
                new_signal = -current_size * 0.7
        
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
            # Exit long if 4H HMA turns bearish
            if position_side > 0 and hma_4h_bearish:
                trend_reversal = True
            # Exit short if 4H HMA turns bullish
            if position_side < 0 and hma_4h_bullish:
                trend_reversal = True
        
        # === SESSION EXIT (close position outside session) ===
        session_exit = False
        if in_position and not in_session and position_side != 0:
            # Optional: close positions outside high-liquidity session
            # Commented out to allow overnight holds for strong trends
            # session_exit = True
            pass
        
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
                # Same direction, maintain position
                pass
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