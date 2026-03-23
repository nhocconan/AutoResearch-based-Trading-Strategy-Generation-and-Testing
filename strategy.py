#!/usr/bin/env python3
"""
Experiment #085: 1h Primary + 4h/1d HTF — Trend Pullback with Session/Volume Filters

Hypothesis: 1h timeframe with strict confluence filters can beat 4h strategies by catching
entries earlier while maintaining low trade count. Use 4h HMA for trend, 1h RSI(7) for
pullback timing, volume filter, and UTC session filter (8-20) to reduce trades to 30-60/year.

Key innovations:
1) 4h HMA(21) for trend bias — only trade with 4h trend direction
2) 1d HMA(21) slope for macro confirmation — skip trades against daily trend
3) 1h RSI(7) pullback entries — RSI<45 for long, RSI>55 for short (loose enough for trades)
4) Volume filter — only enter when volume > 0.8x 20-bar average
5) Session filter — only enter 8-20 UTC (reduces trades by ~50%)
6) Funding rate z-score contrarian boost when extreme
7) 2.5*ATR trailing stoploss
8) Discrete sizing: 0.25 base, 0.35 max with funding boost

Why this should work:
- 1h entries catch moves earlier than 4h strategies
- Session + volume filters keep trade count low (target 40-60/year)
- 4h + 1d double HTF filter prevents counter-trend trades in 2025 bear market
- RSI(7) is loose enough to generate trades (avoid 0-trade failure)
- Funding contrarian adds edge on BTC/ETH specifically

Position size: 0.25 base, 0.35 max
Stoploss: 2.5*ATR trailing
Target: 40-60 trades/year, Sharpe > 0.5
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_pullback_4h1d_hma_session_vol_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_ema(close, period=21):
    """Calculate EMA."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_funding_zscore(prices, symbol, lookback=30):
    """Calculate funding rate z-score for contrarian signal."""
    try:
        import os
        funding_path = f"data/processed/funding/{symbol.lower()}.parquet"
        if os.path.exists(funding_path):
            df_funding = pd.read_parquet(funding_path)
            if 'funding_rate' in df_funding.columns:
                funding = df_funding['funding_rate'].values
                min_len = min(len(funding), len(prices))
                funding = funding[-min_len:]
                funding_s = pd.Series(funding)
                funding_mean = funding_s.rolling(window=lookback, min_periods=lookback).mean()
                funding_std = funding_s.rolling(window=lookback, min_periods=lookback).std()
                zscore = (funding_s - funding_mean) / (funding_std + 1e-10)
                zscore = zscore.fillna(0.0).values
                if len(zscore) < len(prices):
                    pad = np.zeros(len(prices) - len(zscore))
                    zscore = np.concatenate([pad, zscore])
                return zscore[:len(prices)]
    except Exception:
        pass
    return np.zeros(len(prices))

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    return (open_time // (1000 * 60 * 60)) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Extract symbol
    symbol = "BTCUSDT"
    if hasattr(prices, 'attrs') and 'symbol' in prices.attrs:
        symbol = prices.attrs['symbol']
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for trend
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d HMA for macro trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1d HMA slope
    hma_1d_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1d_aligned[i]) and not np.isnan(hma_1d_aligned[i-1]) and hma_1d_aligned[i-1] != 0:
            hma_1d_slope[i] = (hma_1d_aligned[i] - hma_1d_aligned[i-1]) / hma_1d_aligned[i-1] * 100
        else:
            hma_1d_slope[i] = 0.0
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    ema_21 = calculate_ema(close, period=21)
    ema_50 = calculate_ema(close, period=50)
    
    # Volume average (20 bars)
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=20, min_periods=20).mean().values
    
    # Funding z-score
    funding_z = calculate_funding_zscore(prices, symbol, lookback=30)
    
    # UTC hour for session filter
    utc_hours = np.array([get_utc_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.35
    
    # Track position state
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_7[i]) or np.isnan(ema_21[i]):
            continue
        if atr_14[i] == 0 or np.isnan(atr_14[i]):
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === MACRO TREND CONFIRMATION (1d HMA slope) ===
        daily_bullish = hma_1d_slope[i] > 0.0
        daily_bearish = hma_1d_slope[i] < 0.0
        
        # === VOLUME FILTER ===
        vol_ok = volume[i] > 0.8 * vol_avg[i] if not np.isnan(vol_avg[i]) else True
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= utc_hours[i] <= 20
        
        # === FUNDING CONTRARIAN ===
        funding_extreme_long = funding_z[i] > 2.0
        funding_extreme_short = funding_z[i] < -2.0
        
        # === RSI PULLBACK SIGNALS (loose enough for trades) ===
        rsi_pullback_long = rsi_7[i] < 45.0
        rsi_pullback_short = rsi_7[i] > 55.0
        rsi_oversold = rsi_7[i] < 30.0
        rsi_overbought = rsi_7[i] > 70.0
        
        # === EMA CONFIRMATION ===
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Only enter during session hours
        if in_session and vol_ok:
            # --- LONG ENTRY: 4h uptrend + daily confirms + RSI pullback ---
            if price_above_hma_4h and daily_bullish:
                # Strong long: 4h bullish + daily bullish + RSI pullback + EMA bullish
                if rsi_pullback_long and ema_bullish:
                    new_signal = POSITION_SIZE_BASE
                    if funding_extreme_short:
                        new_signal = POSITION_SIZE_MAX
                # Weak long: 4h bullish + RSI very oversold
                elif rsi_oversold:
                    new_signal = POSITION_SIZE_BASE
            
            # --- SHORT ENTRY: 4h downtrend + daily confirms + RSI pullback ---
            if price_below_hma_4h and daily_bearish:
                # Strong short: 4h bearish + daily bearish + RSI pullback + EMA bearish
                if rsi_pullback_short and ema_bearish:
                    new_signal = -POSITION_SIZE_BASE
                    if funding_extreme_long:
                        new_signal = -POSITION_SIZE_MAX
                # Weak short: 4h bearish + RSI very overbought
                elif rsi_overbought:
                    new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION (allow holding outside session) ===
        if in_position and new_signal == 0.0:
            if position_side > 0 and rsi_14[i] < 70.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and rsi_14[i] > 30.0:
                new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND CHANGE ===
        if in_position and position_side > 0:
            if price_below_hma_4h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_4h:
                new_signal = 0.0
        
        # === EXIT ON RSI EXTREME (take profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals